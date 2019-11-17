#pragma once

#include "HorseIR/Tree/Tree.h"
#include "PTX/Program.h"

#include "Runtime/Runtime.h"

namespace Runtime {

class GPUCompiler
{
public:
	GPUCompiler(Runtime& runtime) : m_runtime(runtime) {}

	PTX::Program *Compile(const HorseIR::Program *program) const;

	void Optimize(PTX::Program *program) const;

private:
	Runtime& m_runtime;
};

}
