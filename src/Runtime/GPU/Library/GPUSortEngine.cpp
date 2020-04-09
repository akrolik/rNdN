#include "Runtime/GPU/Library/GPUSortEngine.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Runtime/Interpreter.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ConstantBuffer.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"

#include "Utils/Math.h"

namespace Runtime {

std::pair<TypedVectorBuffer<std::int64_t> *, DataBuffer *> GPUSortEngine::Sort(const std::vector<DataBuffer *>& arguments)
{
	// Get the execution engine for the init/sort functions

	Interpreter interpreter(m_runtime);

	// Initialize the index buffer and sort buffer padding

	auto initFunction = BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(0))->GetFunction();
	auto initBuffers = interpreter.Execute(initFunction, {arguments.at(2), arguments.at(3)});

	auto indexBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(initBuffers.at(0));
	auto dataBuffer = initBuffers.at(1);

	// Perform the iterative sort

	auto sortFunction = BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(1))->GetFunction();

	// Compute the active size and iterations

	const auto activeSize = indexBuffer->GetElementCount();
	const auto iterations = static_cast<unsigned int>(std::log2(activeSize));

	for (auto stage = 0u; stage < iterations; ++stage)
	{
		for (auto substage = 0u; substage <= stage; ++substage)
		{
			// Collect the input bufers for sorting: (index, data), order, stage, substage

			std::vector<DataBuffer *> sortBuffers(initBuffers);
			auto stageBuffer = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, stage);
			auto substageBuffer = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, substage);

			sortBuffers.push_back(arguments.at(3));
			sortBuffers.push_back(stageBuffer);
			sortBuffers.push_back(substageBuffer);

			// Execute!

			interpreter.Execute(sortFunction, sortBuffers);

			delete stageBuffer;
			delete substageBuffer;
		}
	}

	// Resize the index buffer to fit the number of indices if needed

	auto size = 0;
	if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(2), false))
	{
		size = vectorBuffer->GetElementCount();
	}
	else if (auto listBuffer = BufferUtils::GetBuffer<ListBuffer>(arguments.at(2), false))
	{
		size = BufferUtils::GetBuffer<VectorBuffer>(listBuffer->GetCell(0))->GetElementCount();
	}
	else
	{
		Utils::Logger::LogError("GPU sort requires vector or list data, received " + arguments.at(2)->Description());
	}

	if (activeSize > size)
	{
		// Allocate a smaller buffer for each column and copy the data

		auto resizedIndexBuffer = VectorBuffer::CreateEmpty(indexBuffer->GetType(), new Analysis::Shape::ConstantSize(size));
		CUDA::Buffer::Copy(resizedIndexBuffer->GetGPUWriteBuffer(), indexBuffer->GetGPUReadBuffer(), resizedIndexBuffer->GetGPUBufferSize());

		Utils::Logger::LogDebug("Resized buffer [" + indexBuffer->Description() + "] to [" + resizedIndexBuffer->Description() + "]");
		delete indexBuffer;

		//TODO: Centralize resize code
		if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(dataBuffer, false))
		{
			auto resizedVectorBuffer = VectorBuffer::CreateEmpty(vectorBuffer->GetType(), new Analysis::Shape::ConstantSize(size));
			CUDA::Buffer::Copy(resizedVectorBuffer->GetGPUWriteBuffer(), vectorBuffer->GetGPUReadBuffer(), resizedVectorBuffer->GetGPUBufferSize());

			Utils::Logger::LogDebug("Resized buffer [" + vectorBuffer->Description() + "] to [" + resizedVectorBuffer->Description() + "]");
			delete vectorBuffer;

			return {BufferUtils::GetVectorBuffer<std::int64_t>(resizedIndexBuffer), resizedVectorBuffer};
		}
		else if (auto listBuffer = BufferUtils::GetBuffer<ListBuffer>(dataBuffer, false))
		{
			std::vector<DataBuffer *> resizedCellBuffers;
			for (const auto cellBuffer : listBuffer->GetCells())
			{
				auto vectorCellBuffer = BufferUtils::GetBuffer<VectorBuffer>(cellBuffer);

				auto resizedCellBuffer = VectorBuffer::CreateEmpty(vectorCellBuffer->GetType(), new Analysis::Shape::ConstantSize(size));
				CUDA::Buffer::Copy(resizedCellBuffer->GetGPUWriteBuffer(), vectorCellBuffer->GetGPUReadBuffer(), resizedCellBuffer->GetGPUBufferSize());

				resizedCellBuffers.push_back(resizedCellBuffer);
			}

			auto resizedListBuffer = new ListBuffer(resizedCellBuffers);
			Utils::Logger::LogDebug("Resized buffer [" + listBuffer->Description() + "] to [" + resizedListBuffer->Description() + "]");

			return {BufferUtils::GetVectorBuffer<std::int64_t>(resizedIndexBuffer), resizedListBuffer};
		}
		else
		{
			Utils::Logger::LogError("Unable to resize buffer " + dataBuffer->Description());
		}
	}

	return {indexBuffer, dataBuffer};
}

}
