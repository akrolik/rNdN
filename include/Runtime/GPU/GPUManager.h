#pragma once

#include "CUDA/Device.h"
#include "CUDA/ExternalModule.h"
#include "CUDA/Platform.h"

#include "Runtime/GPU/GPUProgram.h"

namespace Runtime {

class GPUManager
{
public:
	GPUManager() {}
	GPUManager(GPUManager const&) = delete;
	void operator=(GPUManager const&) = delete;

	void Initialize();

	std::unique_ptr<CUDA::Device>& GetCurrentDevice();

	const std::vector<CUDA::ExternalModule>& GetExternalModules() const { return m_externalModules; }

	void SetProgram(const GPUProgram *program) { m_program = program; }
	const GPUProgram *GetProgram() const { return m_program; }

private:
	void InitializeCUDA();
	void InitializeLibraries();

	CUDA::Platform m_platform;
	std::vector<CUDA::ExternalModule> m_externalModules;

	const GPUProgram *m_program = nullptr;
};

}
