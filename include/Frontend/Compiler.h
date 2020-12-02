#pragma once

#include "HorseIR/Tree/Tree.h"
#include "PTX/Tree/Tree.h"

#include "CUDA/Device.h"

namespace Frontend {

class Compiler
{
public:
	Compiler(std::unique_ptr<CUDA::Device>& device) : m_device(device) {}

	PTX::Program *Compile(const HorseIR::Program *program) const;

	void Optimize(PTX::Program *program) const;

private:
	std::unique_ptr<CUDA::Device>& m_device;
};

}
