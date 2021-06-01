#include "Runtime/GPU/Library/HashMemberEngine.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/ConstantBuffer.h"
#include "Runtime/GPU/ExecutionEngine.h"

#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace Runtime {
namespace GPU {

VectorBuffer *HashMemberEngine::Member(const std::vector<const DataBuffer *>& arguments)
{
	// Arguments:
	//   0 - Hash function
	//   1 - Member function
	//   2 - Left data
	//   3 - Right data

	// Get the execution engine for the hash/member functions

	ExecutionEngine engine(m_runtime, m_program);

	// Construct the hash table

	auto leftBuffer = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(2));
	auto rightBuffer = BufferUtils::GetBuffer<VectorBuffer>(arguments.at(3));

	auto size = rightBuffer->GetElementCount();

	const auto shift = Utils::Options::GetAlgorithm_HashSize();
	const auto powerSize = Utils::Math::Power2(size) << shift;
	const auto hashSize = new TypedConstantBuffer<std::int32_t>(HorseIR::BasicType::BasicKind::Int32, powerSize);

	auto hashFunction = GetFunction(arguments.at(0));
	auto hashBuffers = engine.Execute(hashFunction, {rightBuffer, hashSize});

	auto keysBuffer = hashBuffers.at(0);

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Hash table keys: " + keysBuffer->DebugDump());
	}

	// Perform the member match

	auto memberFunction = GetFunction(arguments.at(1));
	auto memberBuffers = engine.Execute(memberFunction, {keysBuffer, leftBuffer});

	return BufferUtils::GetBuffer<VectorBuffer>(memberBuffers.at(0));
}

}
}
