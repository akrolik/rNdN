#include "GPUUtil/CUDAModule.h"

#include "GPUUtil/CUDAUtils.h"

void CUDAModule::AddKernel(CUDAKernel &kernel)
{
	checkDriverResult(cuModuleLoadData(&m_module, kernel.GetBinary()));
	checkDriverResult(cuModuleGetFunction(&kernel.GetKernel(), m_module, kernel.GetName().c_str()));

	// m_kernels.push_back(kernel);
}
