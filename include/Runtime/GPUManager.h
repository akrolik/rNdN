#pragma once

#include "CUDA/Device.h"
#include "CUDA/ExternalModule.h"
#include "CUDA/Module.h"
#include "CUDA/Platform.h"

#include "PTX/Program.h"

namespace Runtime {

class GPUManager
{
public:
	void Initialize();

	std::unique_ptr<CUDA::Device>& GetCurrentDevice();

	CUDA::Module AssembleProgram(const PTX::Program *program) const;

private:
	void InitializeCUDA();
	void InitializeLibraries();

	CUDA::Platform m_platform;
	std::vector<CUDA::ExternalModule> m_externalModules;
};

}
