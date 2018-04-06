#include "CUDA/Platform.h"

#include <iostream>
#include <iomanip>

#include "CUDA/Utils.h"

namespace CUDA {

void Platform::Initialize()
{
	if (!m_initialized)
	{
		checkDriverResult(cuInit(0));
		std::cout << "CUDA driver initialized" << std::endl;

		LoadDevices();
		m_initialized = true;
	}
}

void Platform::LoadDevices()
{
	int count = 0;
	checkRuntimeError(cudaGetDeviceCount(&count));

	std::cout << "Connected devices (" << count << ")" << std::endl;
	for (int i = 0; i < count; ++i)
	{
		std::unique_ptr<Device> device = std::make_unique<Device>(i);

		float mem_f = float(device->GetMemorySize()) / 1024 / 1024 / 1024;
		std::cout << "[" << i << "] " << device->GetName() << " (" << std::setprecision(3) << mem_f << " GB)" << std::endl;

		m_devices.push_back(std::move(device));
	}
}

int Platform::GetDeviceCount()
{
	return m_devices.size();
}

std::unique_ptr<Device>& Platform::GetDevice(int index)
{
	return m_devices.at(index);
}

void Platform::CreateContext(std::unique_ptr<Device>& device)
{
	checkDriverResult(cuCtxCreate(&m_context, 0, device->GetDevice()));
	checkDriverResult(cuCtxSetCurrent(m_context));
}

}
