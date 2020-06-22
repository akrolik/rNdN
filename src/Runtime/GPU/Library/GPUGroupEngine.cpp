#include "Runtime/GPU/Library/GPUGroupEngine.h"

#include "Runtime/Interpreter.h"
#include "Runtime/DataBuffers/BufferUtils.h"

#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/DataBuffers/ListCellBuffer.h"
#include "Runtime/DataBuffers/ListCompressedBuffer.h"
#include "Runtime/GPU/Library/GPUSortEngine.h"
#include "Runtime/GPU/GPUExecutionEngine.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {

const HorseIR::Function *GPUGroupEngine::GetFunction(const HorseIR::FunctionDeclaration *function) const
{
	if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Definition)
	{
		return static_cast<const HorseIR::Function *>(function);
	}
	Utils::Logger::LogError("GPU group library cannot execute function '" + function->GetName() + "'");
}

DictionaryBuffer *GPUGroupEngine::Group(const std::vector<DataBuffer *>& arguments)
{
	// Perform the sort using the sort engine

	auto timeSort_start = Utils::Chrono::Start("Group sort execution");

	auto isShared = (arguments.size() == 5);

	std::vector<DataBuffer *> sortBuffers;
	sortBuffers.push_back(arguments.at(0)); // Init
	sortBuffers.push_back(arguments.at(1)); // Sort
	if (isShared)
	{
		sortBuffers.push_back(arguments.at(2)); // Sort shared
	}
	sortBuffers.push_back(arguments.at(3 + isShared)); // Data

	GPUSortEngine sortEngine(m_runtime, m_program);
	auto [indexBuffer, dataBuffer] = sortEngine.Sort(sortBuffers);

	Utils::Chrono::End(timeSort_start);

	// Execute the group function

	auto timeGroup_start = Utils::Chrono::Start("Group execution");

	auto groupFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(2 + isShared))->GetFunction());

	GPUExecutionEngine engine(m_runtime, m_program);
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

	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
	{
		Utils::Logger::LogDebug("Group dictionary buffer: [entries = " + std::to_string(keysSize) + "]");

		Utils::Logger::LogDebug("Group keys buffer: " + keysBuffer->DebugDump());
		Utils::Logger::LogDebug("Group values buffer: " + valuesBuffer->DebugDump());
	}

	// Create the dictionary buffer

	auto timeCreate_start = Utils::Chrono::Start("Create dictionary");

	ListBuffer *listBuffer = nullptr;
	if (Utils::Options::Get<bool>(Utils::Options::Opt_Algo_group_compressed))
	{
		// Create the dictionary object with compressed list buffer

		listBuffer = new ListCompressedBuffer(valuesBuffer, indexBuffer);
	}
	else
	{
		// Create the dictionary object with divided list cells

		auto values = valuesBuffer->GetCPUReadBuffer();

		std::vector<DataBuffer *> entryBuffers;
		for (auto entryIndex = 0u; entryIndex < keysSize; ++entryIndex)
		{
			// Compute the index range, spanning the last entry to the end of the data

			auto offset = values->GetValue(entryIndex);
			auto end = ((entryIndex + 1) == keysSize) ? indexBuffer->GetElementCount() : values->GetValue(entryIndex + 1);
			auto size = (end - offset);

			auto entryType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
			auto entryBuffer = new TypedVectorBuffer<std::int64_t>(entryType, size);

			if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
			{
				Utils::Logger::LogDebug("Initializing entry " + std::to_string(entryIndex) + " buffer: [" + entryBuffer->Description() + "]");
			}

			// Copy the index data

			CUDA::Buffer::Copy(entryBuffer->GetGPUWriteBuffer(), indexBuffer->GetGPUReadBuffer(), size * sizeof(std::int64_t), 0, offset * sizeof(std::int64_t));

			entryBuffers.push_back(entryBuffer);
		}

		listBuffer = new ListCellBuffer(entryBuffers);
	}
	auto dictionaryBuffer = new DictionaryBuffer(keysBuffer, listBuffer);

	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
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

}
