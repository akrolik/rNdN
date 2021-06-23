#include "Runtime/GPU/Library/SortEngine.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ConstantBuffer.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/GPU/ExecutionEngine.h"

#include "Utils/Logger.h"
#include "Utils/Math.h"

namespace Runtime {
namespace GPU {

std::pair<TypedVectorBuffer<std::int64_t> *, DataBuffer *> SortEngine::Sort(const std::vector<const DataBuffer *>& arguments)
{
	// Get the execution engine for the init/sort functions

	ExecutionEngine engine(m_runtime, m_program);

	// Initialize the index buffer and sort buffer padding

	auto isShared = BufferUtils::IsBuffer<FunctionBuffer>(arguments.at(2));

	std::vector<const DataBuffer *> initSortBuffers(std::begin(arguments) + 2 + isShared, std::end(arguments));

	auto initFunction = GetFunction(arguments.at(0));
	auto initBuffers = engine.Execute(initFunction, initSortBuffers);

	auto indexBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(initBuffers.at(0));
	auto dataBuffer = initBuffers.at(1);

	// Perform the iterative sort

	auto sortFunction = GetFunction(arguments.at(1));
	auto sortFunctionShared = (isShared) ? GetFunction(arguments.at(2)) : nullptr;

	// Compute the active size and iterations

	const auto activeSize = indexBuffer->GetElementCount();
	const auto iterations = static_cast<unsigned int>(std::log2(activeSize));

	// Invalidate the CPU buffers as the GPU will be sorting in place

	indexBuffer->InvalidateCPU();
	dataBuffer->InvalidateCPU();

	// Iteratively sort!

	for (auto stage = 0u; stage < iterations; ++stage)
	{
		for (auto substage = 0u; substage <= stage; ++substage)
		{
			const auto subsequenceSize = 2 << (stage - substage);

			// Collect the input bufers for sorting: (index, data), [order], stage, substage

			std::vector<const DataBuffer *> sortBuffers(std::begin(initBuffers), std::end(initBuffers));
			if (arguments.size() == (4 + isShared))
			{
				sortBuffers.push_back(arguments.at(3 + isShared));
			}

			auto stageBuffer = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, stage);
			auto substageBuffer = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, substage);

			sortBuffers.push_back(stageBuffer);
			sortBuffers.push_back(substageBuffer);

			// Execute!

			auto sharedSize = 0u;
			if (isShared)
			{
				const auto program = m_runtime.GetGPUManager().GetProgram();
				const auto kernelCode = program->GetKernelCode(sortFunctionShared->GetName());

				sharedSize = std::get<0>(kernelCode->GetRequiredThreads());
			}

			if (subsequenceSize <= sharedSize)
			{
				// If there are less than 1024 threads per block, all substages left for this stage will fit in shared memory

				const auto maxIterations = Utils::Math::Log2(sharedSize);
				const auto stages = (stage > 0) ? 1 : std::min(iterations, maxIterations);

				auto stagesBuffer = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, stages);
				sortBuffers.push_back(stagesBuffer);

				engine.Execute(sortFunctionShared, sortBuffers);

				stage += stages - 1;
				substage = stage + 1;

				delete stagesBuffer;
			}
			else
			{
				engine.Execute(sortFunction, sortBuffers);
			}
			delete stageBuffer;
			delete substageBuffer;
		}
	}

	// Resize buffers

	Utils::ScopedChrono timeResize("Resize buffers");

	auto inputBuffer = arguments.at(2 + isShared);
	if (auto inputVectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(inputBuffer, false))
	{
		auto outputBuffer = BufferUtils::GetBuffer<VectorBuffer>(dataBuffer);
		auto size = inputVectorBuffer->GetElementCount();

		indexBuffer->Resize(size);
		outputBuffer->Resize(size);

		return {BufferUtils::GetVectorBuffer<std::int64_t>(indexBuffer), outputBuffer};
	}
	else if (auto inputListBuffer = BufferUtils::GetBuffer<ListBuffer>(inputBuffer, false))
	{
		auto outputBuffer = BufferUtils::GetBuffer<ListBuffer>(dataBuffer);
		auto size = BufferUtils::GetBuffer<VectorBuffer>(inputListBuffer->GetCell(0))->GetElementCount();

		indexBuffer->Resize(size);
		outputBuffer->ResizeCells(size);

		return {BufferUtils::GetVectorBuffer<std::int64_t>(indexBuffer), outputBuffer};
	}
	else
	{
		Utils::Logger::LogError("GPU sort requires vector or list data, received " + dataBuffer->Description());
	}
}

}
}
