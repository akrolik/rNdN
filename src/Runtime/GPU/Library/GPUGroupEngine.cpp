#include "Runtime/GPU/Library/GPUGroupEngine.h"

#include "Runtime/Interpreter.h"
#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/GPU/Library/GPUSortEngine.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {

DictionaryBuffer *GPUGroupEngine::Group(const std::vector<DataBuffer *>& arguments)
{
	// Perform the sort using the sort engine

	auto timeSort_start = Utils::Chrono::Start("Group sort execution");

	std::vector<DataBuffer *> sortBuffers;
	sortBuffers.push_back(arguments.at(0)); // Init
	sortBuffers.push_back(arguments.at(1)); // Sort
	sortBuffers.push_back(arguments.at(3)); // Data

	GPUSortEngine sortEngine(m_runtime, m_program);
	auto [indexBuffer, dataBuffer] = sortEngine.Sort(sortBuffers);

	Utils::Chrono::End(timeSort_start);

	// Execute the group function

	auto groupFunction = BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(2))->GetFunction();

	Interpreter interpreter(m_runtime);
	auto groupBuffers = interpreter.Execute(groupFunction, {indexBuffer, dataBuffer});

	// Generate the shape of the group data

	auto keysBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(groupBuffers.at(0));
	auto valuesBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(groupBuffers.at(1));

	auto keysSize = keysBuffer->GetElementCount();
	auto valuesSize = valuesBuffer->GetElementCount();

	if (keysSize != valuesSize)
	{
		Utils::Logger::LogError("Keys and values size mismatch forming @group dictionary [" + std::to_string(keysSize) + " != " + std::to_string(valuesSize) + "]");
	}

	auto values = valuesBuffer->GetCPUReadBuffer();
	auto& indexes = indexBuffer->GetCPUReadBuffer()->GetValues();

	Utils::Logger::LogDebug("Initializing dictionary buffer: [entries = " + std::to_string(keysSize) + "]");

	auto timeCreate_start = Utils::Chrono::Start("Create dictionary");

	std::vector<DataBuffer *> entryBuffers;
	for (auto entryIndex = 0; entryIndex < keysSize; ++entryIndex)
	{
		// Compute the index range, spanning the last entry to the end of the data

		auto offset = values->GetValue(entryIndex);
		auto end = ((entryIndex + 1) == keysSize) ? indexes.size() : values->GetValue(entryIndex + 1);

		CUDA::Vector<std::int64_t> data;
		data.insert(std::begin(data), std::begin(indexes) + offset, std::begin(indexes) + end);

		auto entryType = new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64);
		auto entryData = new TypedVectorData<std::int64_t>(entryType, std::move(data));
		auto entryBuffer = new TypedVectorBuffer<std::int64_t>(entryData);

		Utils::Logger::LogDebug("Initializing entry " + std::to_string(entryIndex) + " buffer: [" + entryBuffer->Description() + "]");

		entryBuffers.push_back(entryBuffer);
	}

	// Create the dictionary buffer

	auto dictionaryBuffer = new DictionaryBuffer(keysBuffer, new ListBuffer(entryBuffers));

	Utils::Chrono::End(timeCreate_start);

	// Delete all intermediate buffers

	delete valuesBuffer;
	delete indexBuffer;
	delete dataBuffer;

	return dictionaryBuffer;
}

}
