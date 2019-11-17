#pragma once

#include "PTX/Program.h"

#include "Runtime/Runtime.h"
#include "Runtime/GPU/GPUProgram.h"

namespace Runtime {

class GPUAssembler
{
public:
	GPUAssembler(Runtime& runtime) : m_runtime(runtime) {}

	const GPUProgram *Assemble(const PTX::Program *program) const;

private:
	Runtime& m_runtime;
};

}
