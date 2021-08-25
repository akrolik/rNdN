#include "Runtime/GPU/Manager.h"

#include "Runtime/GPU/Assembler.h"

#include "CUDA/libdevice.h"
#include "CUDA/libr3d3.h"
#include "CUDA/Utils.h"

#include "Utils/Chrono.h"
#include "Utils/Options.h"
#include "Utils/Logger.h"

namespace Runtime {
namespace GPU {

void Manager::Initialize()
{
	Utils::Logger::LogSection("Initalizing CUDA", false);

	InitializeCUDA();
	InitializeLibraries();
}

void Manager::InitializeCUDA()
{
	auto timeCUDA_start = Utils::Chrono::Start("CUDA initialization");

 	// Disable cache for CUDA so compile times are accurate. In a production compiler
	// this would be turned off for efficiency

	setenv("CUDA_CACHE_DISABLE", "1", 1);

	if (sizeof(void *) == 4)
	{
		Utils::Logger::LogError("64-bit platform required");
	}

	// Initialize the CUDA platform and the device driver

	m_platform.Initialize();

	// Check to make sure there is at least one detected GPU

	if (m_platform.GetDeviceCount() == 0)
	{
		Utils::Logger::LogError("No GPU devices detected");
	}

	// By default we use the first CUDA capable GPU for computations

	auto& device = GetCurrentDevice();
	device->SetActive();

	// Complete the CUDA initialization by creating a CUDA context for the device

	m_platform.CreateContext(device);

	Utils::Chrono::End(timeCUDA_start);
}

void Manager::InitializeLibraries()
{
	// Load the libdevice library from file, compile it to PTX, and generate a cubin
	// binary file. Doing so at runtime means we can support new versions of
	// libdevice and future compute versions
	//
	// The library will be linked later (if needed). We consider this cost a "setup"
	// cost that is not included in the compile time since the library must only be
	// compiled once

	if (Utils::Options::IsAssembler_LinkExternal())
	{
		auto timeLibrary_start = Utils::Chrono::Start("CUDA external libraries");

		m_externalModules.push_back(CUDA::libdevice::CreateModule(GetCurrentDevice()));

		Utils::Chrono::End(timeLibrary_start);
	}

	// Instantiate the libr3d3 library, used for utility functions. The library
	// will be access separately and not linked to the main program

	auto timer3d3_start = Utils::Chrono::Start("libr3d3 library");

	Assembler assembler(*this);
	m_library = assembler.Assemble(CUDA::libr3d3::CreateProgram(GetCurrentDevice()), true);

	Utils::Chrono::End(timer3d3_start);
}

std::unique_ptr<CUDA::Device>& Manager::GetCurrentDevice()
{
	return m_platform.GetDevice(0);
}

}
}
