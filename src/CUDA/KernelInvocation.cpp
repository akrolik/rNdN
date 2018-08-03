#include "CUDA/KernelInvocation.h"

#include "CUDA/Utils.h"

#include "Utils/Logger.h"

namespace CUDA {

KernelInvocation::KernelInvocation(Kernel& kernel) : m_kernel(kernel)
{
	m_parameters = ::operator new(m_kernel.GetParametersCount());
}

KernelInvocation::~KernelInvocation()
{
	::operator delete(m_parameters);
}

void KernelInvocation::SetParameter(unsigned int index, Constant &value)
{
	((void **)m_parameters)[index] = value.GetAddress();
}

void KernelInvocation::SetParameter(unsigned int index, Buffer &buffer)
{
	((void **)m_parameters)[index] = &buffer.GetGPUBuffer();
}

void KernelInvocation::Launch()
{
	checkDriverResult(cuLaunchKernel(
				m_kernel.GetKernel(),
				m_gridX, m_gridY, m_gridZ,
				m_blockX, m_blockY, m_blockZ,
				m_sharedMemorySize, 0, (void **)m_parameters, 0
	));

	Utils::Logger::LogInfo("Kernel '" + m_kernel.GetName() + "' launched");
	Utils::Logger::LogInfo(" - Grid: " + std::to_string(m_gridX) + " x " + std::to_string(m_gridY) + " x " + std::to_string(m_gridZ)); 
	Utils::Logger::LogInfo(" - Block: " + std::to_string(m_blockX) + " x " + std::to_string(m_blockY) + " x " + std::to_string(m_blockZ)); 
	Utils::Logger::LogInfo(" - Dynamic shared memory: " + std::to_string(m_sharedMemorySize) + " bytes");
}

}
