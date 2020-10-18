#pragma once

#include "CUDA/Device.h"
#include "CUDA/ExternalModule.h"
#include "CUDA/Platform.h"

#include "Runtime/GPU/Program.h"

namespace Runtime {
namespace GPU {

class Manager
{
public:
	Manager() {}
	Manager(Manager const&) = delete;
	void operator=(Manager const&) = delete;

	void Initialize();

	std::unique_ptr<CUDA::Device>& GetCurrentDevice();

	const std::vector<CUDA::ExternalModule>& GetExternalModules() const { return m_externalModules; }

	void SetProgram(const Program *program) { m_program = program; }
	const Program *GetProgram() const { return m_program; }

	// libr3d3

	const Program *GetLibrary() const { return m_library; }

private:
	void InitializeCUDA();
	void InitializeLibraries();

	CUDA::Platform m_platform;
	std::vector<CUDA::ExternalModule> m_externalModules;

	const Program *m_program = nullptr;
	const Program *m_library = nullptr;
};

}
}
