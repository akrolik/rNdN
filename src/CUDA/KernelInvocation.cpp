#include "CUDA/KernelInvocation.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace CUDA {

KernelInvocation::KernelInvocation(Kernel& kernel) : m_kernel(kernel)
{
	m_parameters.resize(m_kernel.GetParametersCount());
}

void KernelInvocation::SetParameter(unsigned int index, Constant &value)
{
	m_parameters.at(index) = value.GetAddress();
}

void KernelInvocation::SetParameter(unsigned int index, Buffer &buffer)
{
	m_parameters.at(index) = &buffer.GetGPUBuffer();
}

void KernelInvocation::Launch()
{
	Utils::Logger::LogDebug("Kernel '" + m_kernel.GetName() + "' launched");
	Utils::Logger::LogDebug(" - Grid: " + std::to_string(m_gridX) + " x " + std::to_string(m_gridY) + " x " + std::to_string(m_gridZ)); 
	Utils::Logger::LogDebug(" - Block: " + std::to_string(m_blockX) + " x " + std::to_string(m_blockY) + " x " + std::to_string(m_blockZ)); 
	Utils::Logger::LogDebug(" - Dynamic shared memory: " + std::to_string(m_dynamicSharedMemorySize) + " bytes");

	auto timeExecution_start = Utils::Chrono::StartCUDA("CUDA kernel '" + m_kernel.GetName() + "' execution");
	checkDriverResult(cuLaunchKernel(
				m_kernel.GetKernel(),
				m_gridX, m_gridY, m_gridZ,
				m_blockX, m_blockY, m_blockZ,
				m_dynamicSharedMemorySize, 0, (void **)m_parameters.data(), 0
	));
	Utils::Chrono::End(timeExecution_start);
}

}
