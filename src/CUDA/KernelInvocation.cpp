#include "CUDA/KernelInvocation.h"

#include "CUDA/Chrono.h"
#include "CUDA/Utils.h"

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
	Utils::Logger::LogInfo("Kernel '" + m_kernel.GetName() + "' launched");
	Utils::Logger::LogInfo(" - Grid: " + std::to_string(m_gridX) + " x " + std::to_string(m_gridY) + " x " + std::to_string(m_gridZ)); 
	Utils::Logger::LogInfo(" - Block: " + std::to_string(m_blockX) + " x " + std::to_string(m_blockY) + " x " + std::to_string(m_blockZ)); 
	Utils::Logger::LogInfo(" - Dynamic shared memory: " + std::to_string(m_sharedMemorySize) + " bytes");

	auto start = Chrono::Start();
	checkDriverResult(cuLaunchKernel(
				m_kernel.GetKernel(),
				m_gridX, m_gridY, m_gridZ,
				m_blockX, m_blockY, m_blockZ,
				m_sharedMemorySize, 0, (void **)m_parameters.data(), 0
	));
	auto timeExecution = Chrono::End(start);

	Utils::Logger::LogTiming("CUDA kernel '" + m_kernel.GetName() + "' execution", timeExecution);
}

}
