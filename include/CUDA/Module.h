#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "CUDA/ExternalModule.h"

namespace CUDA {

class Module
{
public:
	void AddLinkedModule(const ExternalModule& module);
	void AddPTXModule(const std::string& code);

	bool IsCompiled() const { return m_binary != nullptr; }
	void Compile();

	const CUmodule& GetModule() const { return m_module; }

private:
	std::vector<std::reference_wrapper<const ExternalModule>> m_linkedModules;
	std::vector<std::string> m_code;

	void *m_binary = nullptr;
	size_t m_binarySize = 0;

	CUmodule m_module;
};

}
