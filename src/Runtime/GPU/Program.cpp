#include "Runtime/GPU/Program.h"

namespace Runtime {
namespace GPU {

CUDA::Kernel Program::GetKernel(const std::string& name) const
{
	return CUDA::Kernel(name, *m_binary);
}

const PTX::FunctionDefinition<PTX::VoidType> *Program::GetKernelCode(const std::string& name) const
{
	return m_program->GetEntryFunction(name);
}

}
}
