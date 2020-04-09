#pragma once

#include <vector>

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class BuiltinExecutionEngine
{
public:
	BuiltinExecutionEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	std::vector<DataBuffer *> Execute(const HorseIR::BuiltinFunction *function, const std::vector<DataBuffer *>& arguments);

private:
	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
