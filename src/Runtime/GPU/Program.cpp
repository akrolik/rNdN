#include "Runtime/GPU/Program.h"

namespace Runtime {
namespace GPU {

CUDA::Kernel Program::GetKernel(const std::string& name) const
{
	return CUDA::Kernel(name, m_binary);
}

const PTX::FunctionOptions& Program::GetKernelOptions(const std::string& name) const
{
	return m_program->GetEntryFunction(name)->GetOptions();
}

}
}
