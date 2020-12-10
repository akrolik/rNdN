#include "CUDA/Kernel.h"

#include <iostream>

#include "CUDA/Module.h"
#include "CUDA/Utils.h"

namespace CUDA {

Kernel::Kernel(const std::string& name, const Module& module) : m_name(name), m_module(module)
{
	checkDriverResult(cuModuleGetFunction(&m_kernel, module.GetModule(), m_name.c_str()));
}

int Kernel::GetProperty(CUfunction_attribute attribute) const
{
	int value;
	checkDriverResult(cuFuncGetAttribute(&value, attribute, m_kernel));
	return value;
}

int Kernel::GetMaxThreads() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

int Kernel::GetSharedBytes() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
}

int Kernel::GetConstBytes() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
}

int Kernel::GetLocalBytes() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
}

int Kernel::GetNumberRegisters() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_NUM_REGS);
}

int Kernel::GetMaxDynamicSharedBytes() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
}

int Kernel::GetPreferredSharedCarveout() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT);
}

int Kernel::GetPTXVersion() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_PTX_VERSION);
}

int Kernel::GetBinaryVersion() const
{
	return GetProperty(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
}

}
