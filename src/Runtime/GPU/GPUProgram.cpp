#include "Runtime/GPU/GPUProgram.h"

namespace Runtime {

CUDA::Kernel GPUProgram::GetKernel(const std::string& name, unsigned int argumentCount) const
{
	return CUDA::Kernel(name, argumentCount, m_binary);
}

const PTX::FunctionOptions& GPUProgram::GetKernelOptions(const std::string& name) const
{
	return m_program->GetEntryFunction(name)->GetOptions();
}

}
