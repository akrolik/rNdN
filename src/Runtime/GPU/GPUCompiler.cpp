#include "Runtime/GPU/GPUCompiler.h"

#include "Analysis/DataObject/DataObjectAnalysis.h"
#include "Analysis/Geometry/KernelOptionsAnalysis.h"
#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"
#include "Analysis/Shape/ShapeAnalysis.h"

#include "Codegen/CodeGenerator.h"
#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

#include "HorseIR/Semantics/SemanticAnalysis.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace Runtime {

PTX::Program *GPUCompiler::Compile(const HorseIR::Program *program) const
{
	auto timeCodegen_start = Utils::Chrono::Start("Codegen");

	// Get the entry function

	auto entry = HorseIR::SemanticAnalysis::GetEntry(program);

	// Collect input/output shapes and data objects interprocedurally

	Analysis::DataObjectAnalysis dataAnalysis(program);
	dataAnalysis.Analyze(entry);

	Analysis::ShapeAnalysis shapeAnalysis(dataAnalysis, program);
	shapeAnalysis.Analyze(entry);

	// Generate 64-bit PTX code from the input HorseIR for the current device

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_ptx))
	{
		Utils::Logger::LogSection("Generating PTX program");
	}

	auto& gpu = m_runtime.GetGPUManager();
	auto& device = gpu.GetCurrentDevice();

	// Collect the device properties for codegen

	Codegen::TargetOptions targetOptions;
	targetOptions.ComputeCapability = device->GetComputeCapability();
	targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
	targetOptions.WarpSize = device->GetWarpSize();

	// Initialize the geometry information for code generation

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_ptx))
	{
		Utils::Logger::LogInfo("Target Options");
		Utils::Logger::LogInfo(targetOptions.ToString(), 1);
	}

	Codegen::CodeGenerator<PTX::Bits::Bits64> codegen(targetOptions);

	// Set the input options for each function

	for (const auto module : program->GetModules())
	{
		for (const auto contents : module->GetContents())
		{
			// Only for kernel functions do we compile

			const auto function = dynamic_cast<const HorseIR::Function *>(contents);
			if (function == nullptr)
			{
				continue;
			}

			if (!function->IsKernel())
			{
				continue;
			}

			// Get the input options for the function

			const auto& localShapeAnalysis = shapeAnalysis.GetAnalysis(function);

			Analysis::KernelOptionsAnalysis optionsAnalysis;
			optionsAnalysis.Analyze(function, localShapeAnalysis);

			auto inputOptions = optionsAnalysis.GetInputOptions();
			codegen.SetInputOptions(function, inputOptions);
		}
	}

	// Generate the program

	auto timePTX_start = Utils::Chrono::Start("PTX generation");

	auto ptxProgram = codegen.Generate(program);

	Utils::Chrono::End(timePTX_start);
	Utils::Chrono::End(timeCodegen_start);

	// Dump the PTX program or JSON string to stdout

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_ptx))
	{
		Utils::Logger::LogInfo("Generated PTX program");
		Utils::Logger::LogInfo(ptxProgram->ToString(0), 0, true, Utils::Logger::NoPrefix);

		for (const auto module : ptxProgram->GetModules())
		{
			for (const auto& [name, kernel] : module->GetEntryFunctions())
			{
				Utils::Logger::LogInfo("Generated kernel '" + name + "' with options");

				const auto& kernelOptions = kernel->GetOptions();
				Utils::Logger::LogInfo(kernelOptions.ToString(), 1);
			}
		}
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

void GPUCompiler::Optimize(PTX::Program *program) const
{
}

}
