#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <iostream>

namespace CUDA {

class Device
{
public:
	Device(int index);

	int GetIndex() { return m_index; }
	CUdevice& GetDevice() { return m_device; }

	std::string GetName() { return std::string(m_properties.name); }
	size_t GetMemorySize() { return m_properties.totalGlobalMem; }

	std::string GetComputeCapability() const
	{
		return "sm_" + std::to_string(m_properties.major) + std::to_string(m_properties.minor);
	}

	void SetActive();

private:
	int m_index;
	CUdevice m_device;

	cudaDeviceProp m_properties;
};

}
