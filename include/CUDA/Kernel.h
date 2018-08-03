#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

namespace CUDA {

class Module;

class Kernel
{
public:
	Kernel(const std::string& name, unsigned int parametersCount, const Module& module);

	std::string GetName() const { return m_name; }
	unsigned int GetParametersCount() const { return m_parametersCount; }

	CUfunction& GetKernel() { return m_kernel; }

private:
	std::string m_name;
	unsigned int m_parametersCount;
	CUfunction m_kernel;

	const Module& m_module;
};

}
