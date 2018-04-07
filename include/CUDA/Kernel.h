#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

namespace CUDA {

class Module;
class Kernel
{
public:
	Kernel(std::string name, unsigned int paramsCount, Module& module);

	std::string GetName() { return m_name; }
	unsigned int GetParamsCount() { return m_paramsCount; }
	CUfunction& GetKernel() { return m_kernel; }

private:
	std::string m_name;
	unsigned int m_paramsCount;
	CUfunction m_kernel;

	Module& m_module;
};

}
