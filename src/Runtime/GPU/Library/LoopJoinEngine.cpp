#include "Runtime/GPU/Library/LoopJoinEngine.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/GPU/ExecutionEngine.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {
namespace GPU {

ListBuffer *LoopJoinEngine::Join(const std::vector<const DataBuffer *>& arguments)
{
	// Get the execution engine for the count/join functions

	ExecutionEngine engine(m_runtime, m_program);

	// Count the number of results for the join

	auto countFunction = GetFunction(arguments.at(0));
	auto countBuffers = engine.Execute(countFunction, {arguments.at(2), arguments.at(3)});

	auto offsetsBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(countBuffers.at(0));
	auto countBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(countBuffers.at(1));

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Join initialization offsets: " + offsetsBuffer->DebugDump());
		Utils::Logger::LogDebug("Join initialization count: " + countBuffer->DebugDump());
	}

	// Perform the actual join

	auto joinFunction = GetFunction(arguments.at(1));
	auto joinBuffers = engine.Execute(joinFunction, {arguments.at(2), arguments.at(3), offsetsBuffer, countBuffer});

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Join indexes: " + joinBuffers.at(0)->DebugDump());
	}

	return BufferUtils::GetBuffer<ListBuffer>(joinBuffers.at(0));
}

}
}
