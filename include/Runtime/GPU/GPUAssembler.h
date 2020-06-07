#pragma once

#include "PTX/Program.h"

#include "Runtime/GPU/GPUManager.h"
#include "Runtime/GPU/GPUProgram.h"

namespace Runtime {

class GPUAssembler
{
public:
	GPUAssembler(GPUManager& gpuManager) : m_gpuManager(gpuManager) {}

	const GPUProgram *Assemble(const PTX::Program *program) const;

private:
	GPUManager& m_gpuManager;
};

}
