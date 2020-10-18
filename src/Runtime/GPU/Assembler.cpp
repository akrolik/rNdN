#include "Runtime/GPU/Assembler.h"

#include "Assembler/Assembler.h"

#include "Utils/Chrono.h"

namespace Runtime {
namespace GPU {

const Program *Assembler::Assemble(const PTX::Program *program) const
{
	// Generate the CUDA module for the program with the program
	// modules and linked external modules (libraries)

	auto timeAssembly_start = Utils::Chrono::Start("CUDA assembly");

	CUDA::Module cModule;
	for (const auto& module : program->GetModules())
	{
		cModule.AddPTXModule(module->ToString(0));
	}

	//TODO: Add PTX module information
	// auto timeTemp = Utils::Chrono::Start("ELF Gen");

	// auto ssprogram = new SASS::Program();
	// auto ssfunction = new SASS::Function("main_1");
	// ssprogram->AddFunction(ssfunction);

	// Assembler::Assembler assembler;
	// auto binary = assembler.Assemble(ssprogram);

	// Utils::Chrono::End(timeTemp);

	// cModule.AddELFModule(*binary);

	for (const auto& module : m_gpuManager.GetExternalModules())
	{
		cModule.AddExternalModule(module);
	}

	// Compile the module and geneate the cuBin

	cModule.Compile();

	auto gpuProgram = new Program(program, cModule);

	Utils::Chrono::End(timeAssembly_start);

	return gpuProgram;
}

}
}
