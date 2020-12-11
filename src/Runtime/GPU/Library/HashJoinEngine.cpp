#include "Runtime/GPU/Library/HashJoinEngine.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ConstantBuffer.h"
#include "Runtime/DataBuffers/FunctionBuffer.h"
#include "Runtime/GPU/ExecutionEngine.h"

#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace Runtime {
namespace GPU {

const HorseIR::Function *HashJoinEngine::GetFunction(const HorseIR::FunctionDeclaration *function) const
{
	if (function->GetKind() == HorseIR::FunctionDeclaration::Kind::Definition)
	{
		return static_cast<const HorseIR::Function *>(function);
	}
	Utils::Logger::LogError("GPU join library cannot execute function '" + function->GetName() + "'");
}

ListBuffer *HashJoinEngine::Join(const std::vector<DataBuffer *>& arguments)
{
	// Arguments:
	//   0 - Hash function
	//   1 - Count function
	//   2 - join function
	//   3 - Left data
	//   4 - Right data

	// Get the execution engine for the count/join functions

	ExecutionEngine engine(m_runtime, m_program);

	// Construct the hash table

	auto size = 0;
	auto leftBuffer = arguments.at(3);
	auto rightBuffer = arguments.at(4);
	if (const auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(leftBuffer, false))
	{
		size = vectorBuffer->GetElementCount();
	}
	else if (const auto listBuffer = BufferUtils::GetBuffer<ListBuffer>(leftBuffer, false))
	{
		const auto cellBuffer = BufferUtils::GetBuffer<VectorBuffer>(listBuffer->GetCell(0));
		size = cellBuffer->GetElementCount();
	}
	else
	{
		Utils::Logger::LogError("GPU join library unsupported buffer type " + leftBuffer->Description());
	}

	const auto shift = Utils::Options::GetAlgorithm_HashSize();
	const auto powerSize = Utils::Math::Power2(size) << shift;
	const auto hashSize = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, powerSize);

	auto hashFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(0))->GetFunction());
	auto hashBuffers = engine.Execute(hashFunction, {leftBuffer, hashSize});

	auto keysBuffer = hashBuffers.at(0);
	auto valuesBuffer = hashBuffers.at(1);

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Hash table keys: " + keysBuffer->DebugDump());
		Utils::Logger::LogDebug("Hash table values: " + valuesBuffer->DebugDump());
	}

	// Count the number of results for the join

	auto countFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(1))->GetFunction());
	auto countBuffers = engine.Execute(countFunction, {keysBuffer, rightBuffer});

	auto offsetsBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(countBuffers.at(0));
	auto countBuffer = BufferUtils::GetVectorBuffer<std::int64_t>(countBuffers.at(1));

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Join initialization offsets: " + offsetsBuffer->DebugDump());
		Utils::Logger::LogDebug("Join initialization count: " + countBuffer->DebugDump());
	}

	// Perform the actual join

	auto joinFunction = GetFunction(BufferUtils::GetBuffer<FunctionBuffer>(arguments.at(2))->GetFunction());
	auto joinBuffers = engine.Execute(joinFunction, {keysBuffer, valuesBuffer, rightBuffer, offsetsBuffer, countBuffer});

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Join indexes: " + joinBuffers.at(0)->DebugDump());
	}

	return BufferUtils::GetBuffer<ListBuffer>(joinBuffers.at(0));
}

}
}
