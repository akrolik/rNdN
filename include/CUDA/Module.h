#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "CUDA/Kernel.h"

namespace CUDA {

class Module
{
public:
	Module(std::string ptx);

	void *GetBinary() { return m_binary; }
	CUmodule& GetModule() { return m_module; }

private:
	std::string m_ptx;

	void *m_binary = nullptr;
	size_t m_binarySize = 0;

	CUmodule m_module;

	void Compile();
};

}
