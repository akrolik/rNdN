#include "GPUUtil/CUDADevice.h"

#include <iostream>

#include "GPUUtil/CUDAUtils.h"

CUDADevice::CUDADevice(CUdevice device, int index) : m_device(device), m_index(index)
{

}

std::string CUDADevice::GetName()
{
	char name[40];
	checkDriverResult(cuDeviceGetName(name, 40, m_device));
	return std::string(name);
}

size_t CUDADevice::GetMemSize()
{
	size_t mem;
	checkDriverResult(cuDeviceTotalMem(&mem, m_device));
	return mem;
}

void CUDADevice::SetActive()
{
	checkRuntimeError(cudaSetDevice(m_index));

	std::cout << "Device " << m_index << " active" << std::endl;
}
