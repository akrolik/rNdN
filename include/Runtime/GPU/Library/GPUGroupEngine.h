#pragma once

#include <vector>

#include "HorseIR/Tree/Tree.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"
#include "Runtime/DataBuffers/DictionaryBuffer.h"

namespace Runtime {

class GPUGroupEngine
{
public:
	GPUGroupEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

	DictionaryBuffer *Group(const std::vector<DataBuffer *>& arguments);

private:
	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
