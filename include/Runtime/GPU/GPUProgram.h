#pragma once

#include <string>

#include "CUDA/Kernel.h"
#include "CUDA/Module.h"

#include "PTX/PTX.h"

namespace Runtime {

class GPUProgram
{
public:
	GPUProgram(const PTX::Program *program, const CUDA::Module& binary) : m_program(program), m_binary(binary) {}

	CUDA::Kernel GetKernel(const std::string& name) const;
	const PTX::FunctionOptions& GetKernelOptions(const std::string& name) const;

private:
	const PTX::Program *m_program = nullptr;
	CUDA::Module m_binary;
};

}
