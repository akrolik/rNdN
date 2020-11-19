#include "Runtime/GPU/BackendCompiler.h"

#include "Backend/CodeGenerator.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {
namespace GPU {

SASS::Program *BackendCompiler::Compile(const PTX::Program *program) const
{
	auto timeCodegen_start = Utils::Chrono::Start("Backend codegen");

	// Generate 64-bit PTX code from the input HorseIR for the current device

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_sass))
	{
		Utils::Logger::LogSection("Generating SASS program");
	}

	// Generate the program

	auto timeSASS_start = Utils::Chrono::Start("SASS generation");

	Backend::CodeGenerator backend;
	auto sassProgram = backend.Generate(program);

	Utils::Chrono::End(timeSASS_start);
	Utils::Chrono::End(timeCodegen_start);

	// Dump the SASS program to stdout

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_sass))
	{
		Utils::Logger::LogInfo("Generated SASS program");
		Utils::Logger::LogInfo(sassProgram->ToString(), 0, true, Utils::Logger::NoPrefix);
	}

	// Optimize the generated SASS program

	if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
	{
		Optimize(sassProgram);
	}

	return sassProgram;
}

void BackendCompiler::Optimize(SASS::Program *program) const
{
}

}
}
