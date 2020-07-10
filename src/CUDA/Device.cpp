#include "CUDA/Device.h"

#include "CUDA/Utils.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace CUDA {

Device::Device(int index) : m_index(index)
{
	checkDriverResult(cuDeviceGet(&m_device, m_index));
	checkRuntimeError(cudaGetDeviceProperties(&m_properties, m_index));
}

void Device::SetActive()
{
	checkRuntimeError(cudaSetDevice(m_index));

	if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
	{
		Utils::Logger::LogDebug("Device " + std::to_string(m_index) + " selected");
	}
}

}
