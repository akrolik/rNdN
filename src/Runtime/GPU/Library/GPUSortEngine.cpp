#include "Runtime/GPU/Library/GPUSortEngine.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ConstantBuffer.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"
#include "Runtime/GPU/GPUExecutionEngine.h"

#include "Utils/Logger.h"
#include "Utils/Math.h"

namespace Runtime {

const HorseIR::Function *GPUSortEngine::GetFunction(const HorseIR::FunctionDeclaration *function) const
{
	if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Definition)
	{
		return static_cast<const HorseIR::Function *>(function);
	}
	Utils::Logger::LogError("GPU sort library cannot execute function '" + function->GetName() + "'");
}

std::pair<TypedVectorBuffer<std::int64_t> *, DataBuffer *> GPUSortEngine::Sort(const std::vector<DataBuffer *>& arguments)
{
	// Get the execution engine for the init/sort functions

	GPUExecutionEngine engine(m_runtime, m_program);

	// Initialize the index buffer and sort buffer padding

	std::vector<DataBuffer *> initSortBuffers(std::begin(arguments) + 2, std::end(arguments));

	auto initFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(0))->GetFunction());
	auto initBuffers = engine.Execute(initFunction, initSortBuffers);

	auto indexBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(initBuffers.at(0));
	auto dataBuffer = initBuffers.at(1);

	// Perform the iterative sort

	auto sortFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(1))->GetFunction());

	// Compute the active size and iterations

	const auto activeSize = indexBuffer->GetElementCount();
	const auto iterations = static_cast<unsigned int>(std::log2(activeSize));

	// Invalidate the CPU buffers as the GPU will be sorting in place

	indexBuffer->InvalidateCPU();
	dataBuffer->InvalidateCPU();

	// Interate sort!

	for (auto stage = 0u; stage < iterations; ++stage)
	{
		for (auto substage = 0u; substage <= stage; ++substage)
		{
			// Collect the input bufers for sorting: (index, data), [order], stage, substage

			auto stageBuffer = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, stage);
			auto substageBuffer = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, substage);

			std::vector<DataBuffer *> sortBuffers(initBuffers);
			if (arguments.size() == 4)
			{
				sortBuffers.push_back(arguments.at(3));
			}

			sortBuffers.push_back(stageBuffer);
			sortBuffers.push_back(substageBuffer);

			// Execute!

			engine.Execute(sortFunction, sortBuffers);

			delete stageBuffer;
			delete substageBuffer;
		}
	}

	// Resize buffers

	Utils::ScopedChrono timeResize("Resize buffers");

	if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(2), false))
	{
		auto outputBuffer = BufferUtils::GetBuffer<VectorBuffer>(dataBuffer);
		auto size = vectorBuffer->GetElementCount();

		indexBuffer->Resize(size);
		outputBuffer->Resize(size);

		return {BufferUtils::GetVectorBuffer<std::int64_t>(indexBuffer), outputBuffer};
	}
	else if (auto listBuffer = BufferUtils::GetBuffer<ListBuffer>(arguments.at(2), false))
	{
		auto outputBuffer = BufferUtils::GetBuffer<ListBuffer>(dataBuffer);
		auto size = BufferUtils::GetBuffer<VectorBuffer>(listBuffer->GetCell(0))->GetElementCount();

		indexBuffer->Resize(size);
		outputBuffer->ResizeCells(size);

		return {BufferUtils::GetVectorBuffer<std::int64_t>(indexBuffer), outputBuffer};
	}
	else
	{
		Utils::Logger::LogError("GPU sort requires vector or list data, received " + arguments.at(2)->Description());
	}
}

}
