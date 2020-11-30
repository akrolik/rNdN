#pragma once

#include "PTX/Tree/Tree.h"

#include "Runtime/GPU/Manager.h"
#include "Runtime/GPU/Program.h"

namespace Runtime {
namespace GPU {

class Assembler
{
public:
	Assembler(Manager& gpuManager) : m_gpuManager(gpuManager) {}

	const Program *Assemble(const PTX::Program *program, bool library = false) const;

private:
	Manager& m_gpuManager;
};

}
}
