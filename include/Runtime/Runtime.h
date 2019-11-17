#pragma once

#include "Runtime/DataRegistry.h"
#include "Runtime/GPU/GPUManager.h"

namespace Runtime {

class Runtime
{
public:
	void Initialize();
	void LoadData();

	GPUManager& GetGPUManager() { return m_gpu; }
	DataRegistry& GetDataRegistry() { return m_dataRegistry; }

private:
	GPUManager m_gpu;
	DataRegistry m_dataRegistry;
};
}
