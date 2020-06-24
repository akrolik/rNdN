#pragma once

#include "Runtime/DataRegistry.h"
#include "Runtime/GPU/GPUManager.h"

namespace Runtime {

class Runtime
{
public:
	static Runtime *GetInstance()
	{
		if (s_instance == nullptr)
		{
			s_instance = new Runtime();
		}
		return s_instance;
	}

	static void Destroy()
	{
		delete s_instance;
	}

	Runtime(Runtime const&) = delete;
	void operator=(Runtime const&) = delete;

	~Runtime();

	void Initialize();
	void LoadData();

	GPUManager& GetGPUManager() { return m_gpu; }
	DataRegistry& GetDataRegistry() { return m_dataRegistry; }

private:
	static Runtime *s_instance;
	Runtime() {}

	GPUManager m_gpu;
	DataRegistry m_dataRegistry;
};

}
