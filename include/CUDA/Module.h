#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "CUDA/ExternalModule.h"
#include "Assembler/ELFBinary.h"

namespace CUDA {

class Module
{
public:
	void AddExternalModule(const ExternalModule& module);
	void AddELFModule(const Assembler::ELFBinary& module);
	void AddPTXModule(const std::string& code);

	bool IsCompiled() const { return m_binary != nullptr; }
	void Compile();

	const CUmodule& GetModule() const { return m_module; }

private:
	std::vector<std::reference_wrapper<const ExternalModule>> m_externalModules;
	std::vector<std::reference_wrapper<const Assembler::ELFBinary>> m_elfModules;
	std::vector<std::string> m_ptxModules;

	void *m_binary = nullptr;
	size_t m_binarySize = 0;

	CUmodule m_module;
};

}
