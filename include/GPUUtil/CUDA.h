#ifndef R3D3_GPUUTIL_CUDA
#define R3D3_GPUUTIL_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "GPUUtil/CUDADevice.h"

class CUDA {
public:
	CUDA();

	void Initialize();

	int GetDeviceCount();
	CUDADevice& GetDevice(int index);

	static void _checkDriverResult(CUresult result, const char *name, int line);
	static void _checkRuntimeError(cudaError_t error, const char *name, int line);

private:
	bool m_initialized = false;

	std::vector<CUDADevice> m_devices;
	void LoadDevices();
};

#endif
