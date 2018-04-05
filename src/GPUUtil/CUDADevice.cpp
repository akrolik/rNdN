#include "GPUUtil/CUDADevice.h"

#include <iostream>

#include "GPUUtil/CUDAUtils.h"

CUDADevice::CUDADevice(int index) : m_index(index)
{
	checkDriverResult(cuDeviceGet(&m_device, m_index));
	checkRuntimeError(cudaGetDeviceProperties(&m_properties, m_index));
}

void CUDADevice::SetActive()
{
	checkRuntimeError(cudaSetDevice(m_index));

	std::cout << "Device " << m_index << " active" << std::endl;
}
