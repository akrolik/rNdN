#include "CUDA/Device.h"

#include <iostream>

#include "CUDA/Utils.h"

namespace CUDA {

Device::Device(int index) : m_index(index)
{
	checkDriverResult(cuDeviceGet(&m_device, m_index));
	checkRuntimeError(cudaGetDeviceProperties(&m_properties, m_index));
}

void Device::SetActive()
{
	checkRuntimeError(cudaSetDevice(m_index));

	std::cout << "Device " << m_index << " active" << std::endl;
}

}
