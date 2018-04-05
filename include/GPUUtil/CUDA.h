#ifndef R3D3_GPUUTIL_CUDA
#define R3D3_GPUUTIL_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>

#include "GPUUtil/CUDADevice.h"

class CUDA {
public:
	void Initialize();

	int GetDeviceCount();
	std::unique_ptr<CUDADevice>& GetDevice(int index);

	void CreateContext(std::unique_ptr<CUDADevice>& device);

	static void _checkDriverResult(CUresult result, const char *name, int line);
	static void _checkRuntimeError(cudaError_t error, const char *name, int line);

private:
	bool m_initialized = false;

	std::vector<std::unique_ptr<CUDADevice>> m_devices;
	void LoadDevices();

	CUcontext m_context = 0;
};

#endif
