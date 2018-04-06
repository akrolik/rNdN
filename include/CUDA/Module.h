#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "CUDA/Kernel.h"

namespace CUDA {

class Module
{
public:
	void AddKernel(Kernel &kernel);

private:
	CUmodule m_module;
	std::vector<Kernel> m_kernels;

};

}
