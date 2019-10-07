#include "Runtime/JITCompiler.h"

#include "Codegen/CodeGenerator.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

PTX::Program *JITCompiler::Compile(const std::vector<const HorseIR::Function *>& functions)
{
	// Generate 64-bit PTX code from the input HorseIR for the current device

	Utils::Logger::LogSection("Generating PTX program");
	Utils::Logger::LogInfo("Target Options");
	Utils::Logger::LogInfo(m_targetOptions.ToString(), 1);
	Utils::Logger::LogInfo("Input Options");
	Utils::Logger::LogInfo(m_inputOptions.ToString(), 1);

	auto timeCode_start = Utils::Chrono::Start();

	auto codegen = new Codegen::CodeGenerator<PTX::Bits::Bits64>(m_targetOptions, m_inputOptions);
	auto ptxProgram = codegen->Generate(functions);

	auto timeCode = Utils::Chrono::End(timeCode_start);

	Utils::Logger::LogTiming("Codegen", timeCode);

	// Dump the PTX program or JSON string to stdout

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_ptx))
	{
		Utils::Logger::LogInfo("Generated PTX program");
		for (const auto& module : ptxProgram->GetModules())
		{
			Utils::Logger::LogInfo(module->ToString(), 0, true, Utils::Logger::NoPrefix);
		}
	}

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_json))
	{
		Utils::Logger::LogInfo("Generated PTX program (JSON)");
		for (const auto& module : ptxProgram->GetModules())
		{
			Utils::Logger::LogInfo(module->ToJSON().dump(4), 0, true, Utils::Logger::NoPrefix);
		}
	}

	return ptxProgram;
}

void JITCompiler::Optimize(PTX::Program *program)
{
}

}
