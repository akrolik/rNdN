#pragma once

#include "PTX/Program.h"
#include "SASS/Program.h"

#include "Runtime/GPU/Manager.h"

namespace Runtime {
namespace GPU {

class BackendCompiler
{
public:
	BackendCompiler(Manager& gpuManager) : m_gpuManager(gpuManager) {}

	SASS::Program *Compile(const PTX::Program *program) const;

	void Optimize(SASS::Program *program) const;

private:
	Manager& m_gpuManager;
};

}
}
