#include "CUDA/Device.h"

#include "CUDA/Utils.h"

#include "Utils/Logger.h"

namespace CUDA {

Device::Device(int index) : m_index(index)
{
	checkDriverResult(cuDeviceGet(&m_device, m_index));
	checkRuntimeError(cudaGetDeviceProperties(&m_properties, m_index));
}

void Device::SetActive()
{
	checkRuntimeError(cudaSetDevice(m_index));

	Utils::Logger::LogInfo("Device " + std::to_string(m_index) + " selected");
}

}
