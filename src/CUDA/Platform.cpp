#include "CUDA/Platform.h"

#include <iomanip>
#include <sstream>

#include "CUDA/Utils.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace CUDA {

void Platform::Initialize()
{
	if (!m_initialized)
	{
		checkDriverResult(cuInit(0));
		if (Utils::Options::IsDebug_Print())
		{
			Utils::Logger::LogDebug("CUDA driver initialized");

			auto version = RuntimeVersion();
			auto major = (version / 1000);
			auto minor = (version % 1000) / 10;
			Utils::Logger::LogDebug("Runtime version: " + std::to_string(major) + "." + std::to_string(minor));
		}

		LoadDevices();
		m_initialized = true;
	}
}

void Platform::LoadDevices()
{
	int count = 0;
	checkRuntimeError(cudaGetDeviceCount(&count));

	for (unsigned int i = 0; i < count; ++i)
	{
		std::unique_ptr<Device> device = std::make_unique<Device>(i);
		m_devices.push_back(std::move(device));
	}

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Found " + std::to_string(count) + " connected devices");

		auto i = 0;
		for (const auto& device : m_devices)
		{
			auto gmem_f = float(device->GetGlobalMemorySize()) / 1024 / 1024 / 1024;
			std::stringstream gstream;
			gstream << std::setprecision(3) << gmem_f;
			auto gmem = "gmem=" + gstream.str() + "GB";

			auto smem_f = float(device->GetSharedMemorySize()) / 1024;
			std::stringstream sstream;
			sstream << std::setprecision(3) << smem_f;
			auto smem = "smem" + sstream.str() + "KB";

			auto regs = "regs=" + std::to_string(device->GetRegisterCount());
			auto mp = "mp=" + std::to_string(device->GetMultiProcessorCount());
			auto deviceString = gmem + "|" + smem + "|" + regs + "|" + mp;

			Utils::Logger::LogDebug("[" + std::to_string(i++) + "] " + device->GetName() + " (" + deviceString + ")");
		}
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

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug("Created CUDA context for device " + std::to_string(device->GetIndex()));
	}
}

Platform::~Platform()
{
	if (m_initialized)
	{
		checkDriverResult(cuCtxDestroy(m_context));
	}
}

}
