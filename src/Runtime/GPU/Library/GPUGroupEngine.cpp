#include "Runtime/GPU/Library/GPUGroupEngine.h"

#include "Runtime/Interpreter.h"
#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/DataBuffers/ListCompressedBuffer.h"
#include "Runtime/GPU/Library/GPUSortEngine.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {

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

	auto groupFunction = BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(2 + isShared))->GetFunction();

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

	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
	{
		Utils::Logger::LogDebug("Group dictionary buffer: [entries = " + std::to_string(keysSize) + "]");
	}

	auto timeCreate_start = Utils::Chrono::Start("Create dictionary");

	auto values = valuesBuffer->GetCPUReadBuffer();

	//TODO: Add option for switching between compressed and cell list

	auto dataAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), keysSize);
	auto sizeAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), keysSize);
	auto sizesBuffer = new TypedVectorBuffer<std::int32_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), keysSize);

	auto indexOffset = indexBuffer->GetGPUReadBuffer()->GetGPUBuffer();
	auto sizeOffset = sizesBuffer->GetGPUReadBuffer()->GetGPUBuffer();

	auto dataAddresses = dataAddressesBuffer->GetCPUWriteBuffer();
	auto sizeAddresses = sizeAddressesBuffer->GetCPUWriteBuffer();
	auto sizes = sizesBuffer->GetCPUWriteBuffer();

	std::vector<DataBuffer *> entryBuffers;
	for (auto entryIndex = 0u; entryIndex < keysSize; ++entryIndex)
	{
		// Compute the index range, spanning the last entry to the end of the data

		auto offset = values->GetValue(entryIndex);
		auto end = ((entryIndex + 1) == keysSize) ? indexBuffer->GetElementCount() : values->GetValue(entryIndex + 1);
		auto size = (end - offset);

		dataAddresses->SetValue(entryIndex, indexOffset + offset * sizeof(std::int64_t));
		sizeAddresses->SetValue(entryIndex, sizeOffset + entryIndex * sizeof(std::int32_t));
		sizes->SetValue(entryIndex, size);
	}

	// Create the dictionary buffer

	auto dictionaryBuffer = new DictionaryBuffer(keysBuffer, new ListCompressedBuffer(dataAddressesBuffer, sizeAddressesBuffer, sizesBuffer, indexBuffer));

	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
	{
		//TODO: Debug dump of dictionary fails
		Utils::Logger::LogDebug(dictionaryBuffer->GetValues()->DebugDump());
	}

	Utils::Chrono::End(timeCreate_start);

	// Delete all intermediate buffers

	delete valuesBuffer;
	delete dataBuffer;

	Utils::Chrono::End(timeGroup_start);

	return dictionaryBuffer;
}

}
