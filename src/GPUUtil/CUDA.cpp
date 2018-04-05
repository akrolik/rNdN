#include "GPUUtil/CUDA.h"

#include <iostream>
#include <iomanip>

#include "GPUUtil/CUDAUtils.h"

void CUDA::Initialize()
{
	if (!m_initialized)
	{
		checkDriverResult(cuInit(0));
		std::cout << "CUDA driver initialized" << std::endl;

		LoadDevices();
		m_initialized = true;
	}
}

void CUDA::LoadDevices()
{
	int count = 0;
	checkRuntimeError(cudaGetDeviceCount(&count));

	std::cout << "Connected devices (" << count << ")" << std::endl;
	for (int i = 0; i < count; ++i)
	{
		std::unique_ptr<CUDADevice> device = std::make_unique<CUDADevice>(i);

		float mem_f = float(device->GetMemorySize()) / 1024 / 1024 / 1024;
		std::cout << "[" << i << "] " << device->GetName() << " (" << std::setprecision(3) << mem_f << " GB)" << std::endl;

		m_devices.push_back(std::move(device));
	}
}

int CUDA::GetDeviceCount()
{
	return m_devices.size();
}

std::unique_ptr<CUDADevice>& CUDA::GetDevice(int index)
{
	return m_devices.at(index);
}

void CUDA::CreateContext(std::unique_ptr<CUDADevice>& device)
{
	checkDriverResult(cuCtxCreate(&m_context, 0, device->GetDevice()));
	checkDriverResult(cuCtxSetCurrent(m_context));
}

void CUDA::_checkDriverResult(CUresult result, const char *file, int line)
{
	if (result == CUDA_SUCCESS)
	{
		return;
	}

	const char *name = nullptr;
	const char *string = nullptr;

	CUresult l_result = cuGetErrorName(result, &name);
	if (l_result == CUDA_ERROR_INVALID_VALUE)
	{
		std::cerr << "[Driver Error] Unknown CUDA error <" << file << ":" << line << ">" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	cuGetErrorString(result, &string);

	std::cerr << "[Driver Error] " << name << " <" << file << ":" << line << ">" << std::endl << string << std::endl;
	std::exit(EXIT_FAILURE);
}

void CUDA::_checkRuntimeError(cudaError_t error, const char *file, int line)
{
	if (error == cudaSuccess)
	{
		return;
	}

	std::cerr << "[Runtime Error] " << cudaGetErrorString(error) << " <" << file << ":" << line << ">" << std::endl;
	std::exit(EXIT_FAILURE);

}
