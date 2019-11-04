#include "Runtime/GPUManager.h"

#include "CUDA/libdevice.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"

namespace Runtime {

void GPUManager::Initialize()
{
	Utils::Logger::LogSection("Initalizing CUDA", false);

	InitializeCUDA();
	InitializeLibraries();
}

void GPUManager::InitializeCUDA()
{
 	// Disable cache for CUDA so compile times are accurate. In a production compiler
	// this would be turned off for efficiency

	setenv("CUDA_CACHE_DISABLE", "1", 1);

	if (sizeof(void *) == 4)
	{
		Utils::Logger::LogError("64-bit platform required");
	}

	// Initialize the CUDA platform and the device driver

	auto timeCUDA_start = Utils::Chrono::Start();

	m_platform.Initialize();

	// Check to make sure there is at least one detected GPU

	if (m_platform.GetDeviceCount() == 0)
	{
		Utils::Logger::LogError("No connected devices detected");
	}

	// By default we use the first CUDA capable GPU for computations

	std::unique_ptr<CUDA::Device>& device = m_platform.GetDevice(0);
	device->SetActive();

	// Complete the CUDA initialization by creating a CUDA context for the device

	m_platform.CreateContext(device);

	auto timeCUDA = Utils::Chrono::End(timeCUDA_start);
}

void GPUManager::InitializeLibraries()
{
	// Load the libdevice library from file, compile it to PTX, and generate a cubin
	// binary file. Doing so at runtime means we can support new versions of
	// libdevice and future compute versions
	//
	// The library will be linked later (if needed). We consider this cost a "setup"
	// cost that is not included in the compile time since the library must only be
	// compiled once

	auto timeLibrary_start = Utils::Chrono::Start();

	m_externalModules.push_back(CUDA::libdevice::CreateModule(*GetCurrentDevice()));

	auto timeLibrary = Utils::Chrono::End(timeLibrary_start);
}

CUDA::Module GPUManager::AssembleProgram(const PTX::Program *program) const
{
	// Generate the CUDA module for the program with the program
	// modules and linked external modules (libraries)

	auto timeJIT_start = Utils::Chrono::Start();

	CUDA::Module cModule;
	for (const auto& module : program->GetModules())
	{
		cModule.AddPTXModule(module->ToString());
	}
	for (const auto& module : m_externalModules)
	{
		cModule.AddLinkedModule(module);
	}

	// Compile the module and geneate the cuBin

	cModule.Compile();

	auto timeJIT = Utils::Chrono::End(timeJIT_start);

	return cModule;
}

std::unique_ptr<CUDA::Device>& GPUManager::GetCurrentDevice()
{
	return m_platform.GetDevice(0);
}

}
