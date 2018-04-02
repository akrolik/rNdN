#ifndef R3D3_GPUUTIL_CUDADEVICE
#define R3D3_GPUUTIL_CUDADEVICE

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

class CUDADevice
{
public:
	CUDADevice(int index, CUdevice device);

	std::string GetName();
	size_t GetMemSize();

	void SetActive();

private:
	int m_index;
	CUdevice m_device;

};

#endif
