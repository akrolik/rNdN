#pragma once

#include "HorseIR/Tree/Tree.h"
#include "PTX/Tree/Tree.h"

#include "Runtime/GPU/Manager.h"

namespace Runtime {
namespace GPU {

class Compiler
{
public:
	Compiler(Manager& gpuManager) : m_gpuManager(gpuManager) {}

	PTX::Program *Compile(const HorseIR::Program *program) const;

private:
	Manager& m_gpuManager;
};

}
}
