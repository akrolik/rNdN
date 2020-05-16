#include "Runtime/GPU/Library/GPUUniqueEngine.h"

#include "Runtime/Interpreter.h"
#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/GPU/Library/GPUSortEngine.h"

#include "Utils/Chrono.h"

namespace Runtime {

TypedVectorBuffer<std::int64_t> *GPUUniqueEngine::Unique(const std::vector<DataBuffer *>& arguments)
{
	// Perform the sort using the sort engine

	auto timeSort_start = Utils::Chrono::Start("Unique sort execution");

	std::vector<DataBuffer *> sortBuffers;
	sortBuffers.push_back(arguments.at(0)); // Init
	sortBuffers.push_back(arguments.at(1)); // Sort
	sortBuffers.push_back(arguments.at(3)); // Data

	GPUSortEngine sortEngine(m_runtime, m_program);
	auto [indexBuffer, dataBuffer] = sortEngine.Sort(sortBuffers);

	Utils::Chrono::End(timeSort_start);

	// Execute the unique function

	auto timeUnique_start = Utils::Chrono::Start("Unique execution");

	auto uniqueFunction = BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(2))->GetFunction();

	Interpreter interpreter(m_runtime);
	auto uniqueBuffers = interpreter.Execute(uniqueFunction, {indexBuffer, dataBuffer});

	// Delete all intermediate buffers

	delete indexBuffer;
	delete dataBuffer;

	Utils::Chrono::End(timeUnique_start);

	return {BufferUtils::GetVectorBuffer<std::int64_t>(uniqueBuffers.at(0))};
}

}
