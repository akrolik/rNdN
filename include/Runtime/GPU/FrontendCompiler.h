#pragma once

#include "HorseIR/Tree/Tree.h"
#include "PTX/Program.h"

#include "Runtime/GPU/Manager.h"

namespace Runtime {
namespace GPU {

class FrontendCompiler
{
public:
	FrontendCompiler(Manager& gpuManager) : m_gpuManager(gpuManager) {}

	PTX::Program *Compile(const HorseIR::Program *program) const;

	void Optimize(PTX::Program *program) const;

private:
	Manager& m_gpuManager;
};

}
}
