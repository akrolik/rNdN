#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

namespace CUDA {

class Device
{
public:
	Device(int index);

	int GetIndex() const { return m_index; }

	std::string GetName() const { return std::string(m_properties.name); }

	size_t GetGlobalMemorySize() const { return m_properties.totalGlobalMem; }
	size_t GetSharedMemorySize() const { return m_properties.sharedMemPerBlock; }
	int GetRegisterCount() const { return m_properties.regsPerBlock; }

	int GetMultiProcessorCount() const { return m_properties.multiProcessorCount; }
	int GetMaxThreadsDimension(unsigned int dim) const { return m_properties.maxThreadsDim[dim]; }
	int GetWarpSize() const { return m_properties.warpSize; }

	CUdevice& GetDevice() { return m_device; }

	int GetComputeMajor() const { return m_properties.major; }
	int GetComputeMinor() const { return m_properties.minor; }
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
