#include "CUDA/Platform.h"

#include <iomanip>
#include <sstream>

#include "CUDA/Utils.h"

#include "Utils/Logger.h"

namespace CUDA {

void Platform::Initialize()
{
	if (!m_initialized)
	{
		checkDriverResult(cuInit(0));
		Utils::Logger::LogInfo("CUDA driver initialized");

		LoadDevices();
		m_initialized = true;
	}
}

void Platform::LoadDevices()
{
	int count = 0;
	checkRuntimeError(cudaGetDeviceCount(&count));

	Utils::Logger::LogInfo("Found " + std::to_string(count) + " connected devices");
	for (unsigned int i = 0; i < count; ++i)
	{
		std::unique_ptr<Device> device = std::make_unique<Device>(i);

		float mem_f = float(device->GetMemorySize()) / 1024 / 1024 / 1024;
		std::stringstream gstream;
		gstream << std::setprecision(3) << mem_f;
		std::string mem = gstream.str();

		float smem_f = float(device->GetSharedMemorySize()) / 1024;
		std::stringstream sstream;
		sstream << std::setprecision(3) << smem_f;
		std::string smem = sstream.str();

		Utils::Logger::LogInfo("[" + std::to_string(i) + "] " + device->GetName() + " (gmem=" + mem + "GB|smem=" + smem + "KB)");

		m_devices.push_back(std::move(device));
	}
}

int Platform::GetDeviceCount() const
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

	Utils::Logger::LogInfo("Created CUDA context for device " + std::to_string(device->GetIndex()));
}

Platform::~Platform()
{
	if (m_initialized)
	{
		checkDriverResult(cuCtxDestroy(m_context));
	}
}

}
