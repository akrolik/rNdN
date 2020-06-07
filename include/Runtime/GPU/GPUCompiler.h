#pragma once

#include "HorseIR/Tree/Tree.h"
#include "PTX/Program.h"

#include "Runtime/GPU/GPUManager.h"

namespace Runtime {

class GPUCompiler
{
public:
	GPUCompiler(GPUManager& gpuManager) : m_gpuManager(gpuManager) {}

	PTX::Program *Compile(const HorseIR::Program *program) const;

	void Optimize(PTX::Program *program) const;

private:
	GPUManager& m_gpuManager;
};

}
