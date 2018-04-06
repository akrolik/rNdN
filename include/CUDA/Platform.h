#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <iostream>

#include "CUDA/Device.h"

namespace CUDA {

class Platform {
public:
	void Initialize();

	int GetDeviceCount();
	std::unique_ptr<Device>& GetDevice(int index);

	void CreateContext(std::unique_ptr<Device>& device);

private:
	bool m_initialized = false;

	std::vector<std::unique_ptr<Device>> m_devices;
	void LoadDevices();

	CUcontext m_context = 0;
};

}
