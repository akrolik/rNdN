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

	int GetIndex() const { return m_index; }

	std::string GetName() const { return std::string(m_properties.name); }
	size_t GetMemorySize() const { return m_properties.totalGlobalMem; }
	size_t GetSharedMemorySize() const { return m_properties.sharedMemPerBlock; }
	int GetMaxThreadsDimension(unsigned int dim) const { return m_properties.maxThreadsDim[dim]; }
	int GetWarpSize() const { return m_properties.warpSize; }

	CUdevice& GetDevice() { return m_device; }

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
