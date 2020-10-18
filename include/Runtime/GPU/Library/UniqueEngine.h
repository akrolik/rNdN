#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {
namespace GPU {

class UniqueEngine
{
public:
	UniqueEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	TypedVectorBuffer<std::int64_t> *Unique(const std::vector<DataBuffer *>& arguments);

private:
	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
}
