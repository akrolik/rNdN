#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/ListBuffer.h"

namespace Runtime {
namespace GPU {

class LoopJoinEngine
{
public:
	LoopJoinEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	ListBuffer *Join(const std::vector<const DataBuffer *>& arguments);

private:
	const HorseIR::Function *GetFunction(const HorseIR::FunctionDeclaration *function) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
}
