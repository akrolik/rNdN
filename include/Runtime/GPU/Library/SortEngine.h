#pragma once

#include <utility>
#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {
namespace GPU {

class SortEngine
{
public:
	SortEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	std::pair<TypedVectorBuffer<std::int64_t> *, DataBuffer *> Sort(const std::vector<DataBuffer *>& arguments);

private:
	const HorseIR::Function *GetFunction(const HorseIR::FunctionDeclaration *function) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
}
