#include "Runtime/GPU/Library/GroupEngine.h"

#include "Runtime/Interpreter.h"
#include "Runtime/DataBuffers/BufferUtils.h"

#include "Runtime/DataBuffers/ListCellBuffer.h"
#include "Runtime/DataBuffers/ListCompressedBuffer.h"
#include "Runtime/GPU/Library/SortEngine.h"
#include "Runtime/GPU/ExecutionEngine.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {
namespace GPU {

DictionaryBuffer *GroupEngine::Group(const std::vector<const DataBuffer *>& arguments)
{
	// Perform the sort using the sort engine

	auto timeSort_start = Utils::Chrono::Start("Group sort execution");

	auto isShared = (arguments.size() == 5);

	std::vector<const DataBuffer *> sortBuffers;
	sortBuffers.push_back(arguments.at(0)); // Init
	sortBuffers.push_back(arguments.at(1)); // Sort
	if (isShared)
	{
		sortBuffers.push_back(arguments.at(2)); // Sort shared
	}
	sortBuffers.push_back(arguments.at(3 + isShared)); // Data

	SortEngine sortEngine(m_runtime, m_program);
	auto [indexBuffer, dataBuffer] = sortEngine.Sort(sortBuffers);

	Utils::Chrono::End(timeSort_start);

	// Execute the group function

	auto timeGroup_start = Utils::Chrono::Start("Group execution");

	auto groupFunction = GetFunction(arguments.at(2 + isShared));

	ExecutionEngine engine(m_runtime, m_program);
	auto groupBuffers = engine.Execute(groupFunction, {indexBuffer, dataBuffer});

	// Generate the shape of the group data

	auto keysBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(groupBuffers.at(0));
	auto valuesBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(groupBuffers.at(1));

	auto keysSize = keysBuffer->GetElementCount();
	auto valuesSize = valuesBuffer->GetElementCount();

	if (keysSize != valuesSize)
	{
		Utils::Logger::LogError("Keys and values size mismatch forming @group dictionary [" + std::to_string(keysSize) + " != " + std::to_string(valuesSize) + "]");
	}

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Group dictionary buffer: [entries = " + std::to_string(keysSize) + "]");

		Utils::Logger::LogDebug("Group keys buffer: " + keysBuffer->DebugDump());
		Utils::Logger::LogDebug("Group values buffer: " + valuesBuffer->DebugDump());
	}

	// Create the dictionary buffer

	auto timeCreate_start = Utils::Chrono::Start("Create dictionary");

	auto listBuffer = ConstructListBuffer(indexBuffer, valuesBuffer);
	auto dictionaryBuffer = new DictionaryBuffer(keysBuffer, listBuffer);

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Group dictionary: " + dictionaryBuffer->DebugDump());
	}

	Utils::Chrono::End(timeCreate_start);

	// Delete all intermediate buffers

	delete valuesBuffer;
	delete dataBuffer;

	Utils::Chrono::End(timeGroup_start);

	return dictionaryBuffer;
}

ListBuffer *GroupEngine::ConstructListBuffer(TypedVectorBuffer<std::int64_t> *indexBuffer, TypedVectorBuffer<std::int64_t> *valuesBuffer) const
{
	switch (Utils::Options::GetAlgorithm_GroupKind())
	{
		case Utils::Options::GroupKind::CompressedGroup:
		{
			// Create the dictionary object with compressed list buffer

			return new ListCompressedBuffer(valuesBuffer, indexBuffer);
		}
		case Utils::Options::GroupKind::CellGroup:
		{
			// Create the dictionary object with divided list cells

			auto values = valuesBuffer->GetCPUReadBuffer();
			auto valuesSize = valuesBuffer->GetElementCount();

			std::vector<DataBuffer *> entryBuffers;
			for (auto entryIndex = 0u; entryIndex < valuesSize; ++entryIndex)
			{
				// Compute the index range, spanning the last entry to the end of the data

				auto offset = values->GetValue(entryIndex);
				auto end = ((entryIndex + 1) == valuesSize) ? indexBuffer->GetElementCount() : values->GetValue(entryIndex + 1);
				auto size = (end - offset);

				auto entryType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
				auto entryBuffer = new TypedVectorBuffer<std::int64_t>(entryType, size);

				if (Utils::Options::IsDebug_Print())
				{
					Utils::Logger::LogDebug("Initializing entry " + std::to_string(entryIndex) + " buffer: [" + entryBuffer->Description() + "]");
				}

				// Copy the index data

				CUDA::Buffer::Copy(entryBuffer->GetGPUWriteBuffer(), indexBuffer->GetGPUReadBuffer(), size * sizeof(std::int64_t), 0, offset * sizeof(std::int64_t));

				entryBuffers.push_back(entryBuffer);
			}

			return new ListCellBuffer(entryBuffers);
		}
	}
	Utils::Logger::LogError("GPU group library cannot create list buffer");
}

}
}
