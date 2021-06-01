#pragma once

#include "HorseIR/Tree/Tree.h"

#include "Runtime/Runtime.h"
#include "Runtime/DataBuffers/DataBuffer.h"

namespace Runtime {
namespace GPU {

class LibraryEngine
{
public:
	LibraryEngine(Runtime& runtime, const HorseIR::Program *program) : m_runtime(runtime), m_program(program) {}

protected:
	const HorseIR::Function *GetFunction(const DataBuffer *buffer) const;

	Runtime& m_runtime;
	const HorseIR::Program *m_program = nullptr;
};

}
}
