#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

namespace CUDA {

class Module;

class Kernel
{
public:
	Kernel(const std::string& name, unsigned int paramsCount, const Module& module);

	std::string GetName() const { return m_name; }
	unsigned int GetParamsCount() const { return m_paramsCount; }

	CUfunction& GetKernel() { return m_kernel; }

private:
	std::string m_name;
	unsigned int m_paramsCount;
	CUfunction m_kernel;

	const Module& m_module;
};

}
