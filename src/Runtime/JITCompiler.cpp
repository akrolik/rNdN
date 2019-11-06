#include "Runtime/JITCompiler.h"

#include "Codegen/CodeGenerator.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

PTX::Program *JITCompiler::Compile(const std::vector<const HorseIR::Function *>& functions, const std::vector<const Codegen::InputOptions *>& inputOptions)
{
	// Generate 64-bit PTX code from the input HorseIR for the current device

	Utils::Logger::LogSection("Generating PTX program");
	Utils::Logger::LogInfo("Target Options");
	Utils::Logger::LogInfo(m_targetOptions.ToString(), 1);

	auto index = 0u;
	for (const auto function : functions)
	{
		Utils::Logger::LogInfo("Input Options: " + function->GetName());
		Utils::Logger::LogInfo(inputOptions.at(index++)->ToString(), 1);
	}

	auto timeCode_start = Utils::Chrono::Start();

	auto codegen = new Codegen::CodeGenerator<PTX::Bits::Bits64>(m_targetOptions);
	auto ptxProgram = codegen->Generate(functions, inputOptions);

	auto timeCode = Utils::Chrono::End(timeCode_start);

	Utils::Logger::LogTiming("Codegen", timeCode);

	// Dump the PTX program or JSON string to stdout

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_ptx))
	{
		Utils::Logger::LogInfo("Generated PTX program");
		Utils::Logger::LogInfo(ptxProgram->ToString(0), 0, true, Utils::Logger::NoPrefix);
	}

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_json))
	{
		Utils::Logger::LogInfo("Generated PTX program (JSON)");
		Utils::Logger::LogInfo(ptxProgram->ToJSON().dump(4), 0, true, Utils::Logger::NoPrefix);
	}

	// Optimize the generated PTX program

	if (Utils::Options::Get<>(Utils::Options::Opt_Optimize))
	{
		Optimize(ptxProgram);
	}

	return ptxProgram;
}

void JITCompiler::Optimize(PTX::Program *program)
{
}

}
