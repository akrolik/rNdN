#include "Runtime/GPU/GPUAssembler.h"

#include "Utils/Chrono.h"

namespace Runtime {

const GPUProgram *GPUAssembler::Assemble(const PTX::Program *program) const
{
	// Generate the CUDA module for the program with the program
	// modules and linked external modules (libraries)

	auto timeAssembly_start = Utils::Chrono::Start("CUDA assembly");

	CUDA::Module cModule;
	for (const auto& module : program->GetModules())
	{
		cModule.AddPTXModule(module->ToString(0));
	}

	for (const auto& module : m_gpuManager.GetExternalModules())
	{
		cModule.AddLinkedModule(module);
	}

	// Compile the module and geneate the cuBin

	cModule.Compile();

	auto gpuProgram = new GPUProgram(program, cModule);

	Utils::Chrono::End(timeAssembly_start);

	return gpuProgram;
}

}
