#pragma once

#include <string>

#include "CUDA/Kernel.h"
#include "CUDA/Module.h"

#include "PTX/Tree/Tree.h"

namespace Runtime {
namespace GPU {

class Program
{
public:
	Program(const PTX::Program *program, const CUDA::Module *binary) : m_program(program), m_binary(binary) {}

	CUDA::Kernel GetKernel(const std::string& name) const;
	const PTX::FunctionDefinition<PTX::VoidType> *GetKernelCode(const std::string& name) const;

private:
	const PTX::Program *m_program = nullptr;
	const CUDA::Module *m_binary = nullptr;
};

}
}
