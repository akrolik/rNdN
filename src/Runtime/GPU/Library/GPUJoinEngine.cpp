#include "Runtime/GPU/Library/GPUJoinEngine.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/GPU/GPUExecutionEngine.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

const HorseIR::Function *GPUJoinEngine::GetFunction(const HorseIR::FunctionDeclaration *function) const
{
	if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Definition)
	{
		return static_cast<const HorseIR::Function *>(function);
	}
	Utils::Logger::LogError("GPU join library cannot execute function '" + function->GetName() + "'");
}

ListBuffer *GPUJoinEngine::Join(const std::vector<DataBuffer *>& arguments)
{
	// Get the execution engine for the count/join functions

	GPUExecutionEngine engine(m_runtime, m_program);

	// Count the number of results for the join

	auto countFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(0))->GetFunction());
	auto countBuffers = engine.Execute(countFunction, {arguments.at(2), arguments.at(3)});

	auto offsetsBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(countBuffers.at(0));
	auto countBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(countBuffers.at(1));

	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
	{
		Utils::Logger::LogDebug("Join initialization offsets: " + offsetsBuffer->DebugDump());
		Utils::Logger::LogDebug("Join initialization count: " + countBuffer->DebugDump());
	}

	// Perform the actual join

	auto joinFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(1))->GetFunction());
	auto joinBuffers = engine.Execute(joinFunction, {arguments.at(2), arguments.at(3), offsetsBuffer, countBuffer});

	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
	{
		Utils::Logger::LogDebug("Join indexes: " + joinBuffers.at(0)->DebugDump());
	}

	return BufferUtils::GetBuffer<ListBuffer>(joinBuffers.at(0));
}

}
