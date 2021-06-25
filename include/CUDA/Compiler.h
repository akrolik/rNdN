#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "CUDA/ExternalModule.h"
#include "Assembler/ELFBinary.h"

namespace CUDA {

class Compiler
{
public:
	// Module data for compilation

	void AddExternalModule(const ExternalModule& module);
	void AddELFModule(const Assembler::ELFBinary& module);
	void AddPTXModule(const std::string& code);
	void AddFileModule(const std::string& file);

	// Compilation

	void Compile();
	bool IsCompiled() const { return (m_binary != nullptr); }

	void *GetBinary() { return m_binary; }
	std::size_t GetBinarySize() const { return m_binarySize; }

private:
	std::vector<std::reference_wrapper<const ExternalModule>> m_externalModules;
	std::vector<std::reference_wrapper<const Assembler::ELFBinary>> m_elfModules;
	std::vector<std::string> m_fileModules;
	std::vector<std::string> m_ptxModules;

	void *m_binary = nullptr;
	std::size_t m_binarySize = 0;
};

}
