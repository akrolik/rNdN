#include "Runtime/GPU/Assembler.h"

#include "Assembler/Assembler.h"
#include "Assembler/ELFGenerator.h"

#include "Backend/Compiler.h"

#include "CUDA/Compiler.h"
#include "CUDA/Module.h"

#include "PTX/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#include <fstream>

namespace Runtime {
namespace GPU {

const Program *Assembler::Assemble(PTX::Program *program, bool library) const
{
	// Generate the CUDA module for the program with modules and linked external modules (libraries)

	auto timeAssembler_start = Utils::Chrono::Start("Assembler");

	void *binary = nullptr;
	std::size_t binarySize = 0;

	auto& device = m_gpuManager.GetCurrentDevice();

	if (library)
	{
		// Library modules are compiled with ptxas

		CUDA::Compiler compiler;
		for (const auto& module : program->GetModules())
		{
			compiler.AddPTXModule(PTX::PrettyPrinter::PrettyString(module));
		}
		compiler.Compile(device);

		binary = compiler.GetBinary();
		binarySize = compiler.GetBinarySize();
	}
	else if (Utils::Options::IsAssembler_LoadELF())
	{
		// Load the data

		auto stream = std::ifstream(Utils::Options::GetAssembler_LoadELFFile(), std::ios::in | std::ios::binary);
		std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(stream), {});

		// Copy to new buffer

		binarySize = buffer.size();
		binary = malloc(binarySize);
		std::memcpy(binary, buffer.data(), binarySize);

		stream.close();
	}
	else
	{
		// JIT code is compiled with ptxas or r3d3

		CUDA::Compiler compiler;
		if (Utils::Options::IsBackend_LoadELF())
		{
			compiler.AddFileModule(Utils::Options::GetBackend_LoadELFFile());
		}
		else
		{
			for (const auto& module : program->GetModules())
			{
				switch (Utils::Options::GetBackend_Kind())
				{
					case Utils::Options::BackendKind::ptxas:
					{
						compiler.AddPTXModule(PTX::PrettyPrinter::PrettyString(module));
						break;
					}
					case Utils::Options::BackendKind::r3d3:
					{
						// Generate SASS for PTX program

						Backend::Compiler backendCompiler;
						auto sassProgram = backendCompiler.Compile(program);

						// Generate ELF binrary

						auto timeBinary_start = Utils::Chrono::Start("Binary generator");

						::Assembler::Assembler assembler;
						auto binaryProgram = assembler.Assemble(sassProgram);

						::Assembler::ELFGenerator elfGenerator;
						auto elfProgram = elfGenerator.Generate(binaryProgram);

						Utils::Chrono::End(timeBinary_start);

						// Add finalized ELF binary to the module for linking

						compiler.AddELFModule(*elfProgram);

						if (Utils::Options::IsBackend_SaveELF())
						{
							auto stream = std::ofstream(Utils::Options::GetBackend_SaveELFFile(), std::ios::out | std::ios::binary);
							stream.write((char*)elfProgram->GetData(), elfProgram->GetSize());
							stream.close();
						}
						break;
					}
				}
			}
		}

		// Add external libraries

		if (Utils::Options::IsAssembler_LinkExternal())
		{
			for (const auto& module : m_gpuManager.GetExternalModules())
			{
				compiler.AddExternalModule(module);
			}
		}

		// Compile!

		compiler.Compile(device);

		binary = compiler.GetBinary();
		binarySize = compiler.GetBinarySize();
	}

	if (Utils::Options::IsAssembler_SaveELF())
	{
		auto stream = std::ofstream(Utils::Options::GetAssembler_SaveELFFile(), std::ios::out | std::ios::binary);
		stream.write((char*)binary, binarySize);
		stream.close();
	}

	// Load the binary onto the GPU

	auto gpuModule = new CUDA::Module(binary, binarySize);
	auto gpuProgram = new Program(program, gpuModule);

	Utils::Chrono::End(timeAssembler_start);

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogDebug(gpuProgram->ToString());
	}

	return gpuProgram;
}

}
}
