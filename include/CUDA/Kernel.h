#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

namespace CUDA {

class Module;

class Kernel
{
public:
	Kernel(const std::string& name, const Module& module);

	std::string GetName() const { return m_name; }
	CUfunction& GetKernel() { return m_kernel; }

	// Properties

	int GetMaxThreads() const;

	int GetSharedBytes() const;
	int GetConstBytes() const;
	int GetLocalBytes() const;
	int GetNumberRegisters() const;
	int GetMaxDynamicSharedBytes() const;
	int GetPreferredSharedCarveout() const;

	int GetPTXVersion() const;
	int GetBinaryVersion() const;

private:
	int GetProperty(CUfunction_attribute attribute) const;

	std::string m_name;
	CUfunction m_kernel;

	const Module& m_module;
};

}
