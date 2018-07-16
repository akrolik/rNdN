#include "CUDA/Kernel.h"

#include <iostream>

#include "CUDA/Module.h"
#include "CUDA/Utils.h"

namespace CUDA {

Kernel::Kernel(const std::string& name, unsigned int paramsCount, const Module& module) : m_name(name), m_paramsCount(paramsCount), m_module(module)
{
	checkDriverResult(cuModuleGetFunction(&m_kernel, module.GetModule(), m_name.c_str()));
}

}
