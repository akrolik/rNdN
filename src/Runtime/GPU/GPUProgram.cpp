#include "Runtime/GPU/GPUProgram.h"

namespace Runtime {

CUDA::Kernel GPUProgram::GetKernel(const std::string& name) const
{
	return CUDA::Kernel(name, m_binary);
}

const PTX::FunctionOptions& GPUProgram::GetKernelOptions(const std::string& name) const
{
	return m_program->GetEntryFunction(name)->GetOptions();
}

}
