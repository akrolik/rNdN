#include "Runtime/GPU/Assembler.h"

#include "Assembler/Assembler.h"
#include "Assembler/ELFGenerator.h"
#include "Backend/Compiler.h"

#include "PTX/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Options.h"

namespace Runtime {
namespace GPU {

const Program *Assembler::Assemble(PTX::Program *program, bool library) const
{
	// Generate the CUDA module for the program with the program
	// modules and linked external modules (libraries)

	auto timeAssembler_start = Utils::Chrono::Start("Assembler");

	CUDA::Module cModule;
	for (const auto& module : program->GetModules())
	{
		if (library)
		{
			cModule.AddPTXModule(PTX::PrettyPrinter::PrettyString(module));
		}
		else
		{
			switch (Utils::Options::GetBackend_Kind())
			{
				case Utils::Options::BackendKind::ptxas:
				{
					cModule.AddPTXModule(PTX::PrettyPrinter::PrettyString(module));
					break;
				}
				case Utils::Options::BackendKind::r3d3:
				{
					// Generate SASS for PTX program

					auto& device = m_gpuManager.GetCurrentDevice();
					auto compute = device->GetComputeMajor() * 10 + device->GetComputeMinor();

					if (compute < SASS::COMPUTE_MIN || compute > SASS::COMPUTE_MAX)
					{
						Utils::Logger::LogError("Unsupported CUDA compute capability " + device->GetComputeCapability());
					}

					Backend::Compiler compiler;
					auto sassProgram = compiler.Compile(program);
					sassProgram->SetComputeCapability(compute);

					// Generate ELF binrary

					auto timeBinary_start = Utils::Chrono::Start("Binary generator");

					::Assembler::Assembler assembler;
					auto binaryProgram = assembler.Assemble(sassProgram);

					::Assembler::ELFGenerator elfGenerator;
					auto elfProgram = elfGenerator.Generate(binaryProgram);

					Utils::Chrono::End(timeBinary_start);

					// Add finalized ELF binary to the module for linking

					cModule.AddELFModule(*elfProgram);
					break;
				}
			}
		}
	}

	if (Utils::Options::IsLink_External())
	{
		for (const auto& module : m_gpuManager.GetExternalModules())
		{
			cModule.AddExternalModule(module);
		}
	}

	// Compile the module and geneate the cuBin

	cModule.Compile();

	auto gpuProgram = new Program(program, cModule);

	Utils::Chrono::End(timeAssembler_start);

	return gpuProgram;
}

}
}
