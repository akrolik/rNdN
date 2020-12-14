#include "Runtime/GPU/Library/UniqueEngine.h"

#include "Runtime/Interpreter.h"
#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/GPU/Library/SortEngine.h"

#include "Utils/Chrono.h"

namespace Runtime {
namespace GPU {

TypedVectorBuffer<std::int64_t> *UniqueEngine::Unique(const std::vector<const DataBuffer *>& arguments)
{
	// Perform the sort using the sort engine

	auto timeSort_start = Utils::Chrono::Start("Unique sort execution");

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

	// Execute the unique function

	auto timeUnique_start = Utils::Chrono::Start("Unique execution");

	auto uniqueFunction = BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(2 + isShared))->GetFunction();

	Interpreter interpreter(m_runtime);
	auto uniqueBuffers = interpreter.Execute(uniqueFunction, {indexBuffer, dataBuffer});

	// Delete all intermediate buffers

	delete indexBuffer;
	delete dataBuffer;

	Utils::Chrono::End(timeUnique_start);

	return {BufferUtils::GetVectorBuffer<std::int64_t>(uniqueBuffers.at(0))};
}

}
}
