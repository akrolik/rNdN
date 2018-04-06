#include "CUDA/Module.h"

#include "CUDA/Utils.h"

namespace CUDA {

void Module::AddKernel(Kernel &kernel)
{
	checkDriverResult(cuModuleLoadData(&m_module, kernel.GetBinary()));
	checkDriverResult(cuModuleGetFunction(&kernel.GetKernel(), m_module, kernel.GetName().c_str()));

	// m_kernels.push_back(kernel);
}

}
