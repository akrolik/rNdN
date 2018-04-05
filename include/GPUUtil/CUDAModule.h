#ifndef R3D3_GPUUTIL_CUDAMODULE
#define R3D3_GPUUTIL_CUDAMODULE

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "GPUUtil/CUDAKernel.h"

class CUDAModule
{
public:
	void AddKernel(CUDAKernel &kernel);

private:
	CUmodule m_module;
	std::vector<CUDAKernel> m_kernels;

};

#endif
