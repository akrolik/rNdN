#include "Runtime/GPU/FrontendCompiler.h"

#include "HorseIR/Analysis/DataObject/DataObjectAnalysis.h"
#include "HorseIR/Analysis/Geometry/KernelOptionsAnalysis.h"
#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"
#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"

#include "Codegen/CodeGenerator.h"
#include "Codegen/CodegenOptions.h"
#include "Codegen/InputOptions.h"
#include "Codegen/TargetOptions.h"

#include "HorseIR/Semantics/SemanticAnalysis.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace Runtime {
namespace GPU {

PTX::Program *FrontendCompiler::Compile(const HorseIR::Program *program) const
{
	auto timeCodegen_start = Utils::Chrono::Start("Codegen");

	// Get the entry function

	auto entry = HorseIR::SemanticAnalysis::GetEntry(program);

	// Collect input/output shapes and data objects interprocedurally

	HorseIR::Analysis::DataObjectAnalysis dataAnalysis(program);
	dataAnalysis.Analyze(entry);

	HorseIR::Analysis::ShapeAnalysis shapeAnalysis(dataAnalysis, program);
	shapeAnalysis.Analyze(entry);

	// Generate 64-bit PTX code from the input HorseIR for the current device

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_ptx))
	{
		Utils::Logger::LogSection("Generating PTX program");
	}

	auto& device = m_gpuManager.GetCurrentDevice();

	// Collect the device properties for codegen

	Codegen::CodegenOptions codegenOptions;
	Codegen::TargetOptions targetOptions;
	targetOptions.ComputeCapability = device->GetComputeCapability();
	targetOptions.MaxBlockSize = device->GetMaxThreadsDimension(0);
	targetOptions.WarpSize = device->GetWarpSize();
	targetOptions.SharedMemorySize = device->GetSharedMemorySize();

	// Initialize the geometry information for code generation

	if (Utils::Options::Get<>(Utils::Options::Opt_Print_ptx))
	{
		Utils::Logger::LogInfo("Codegen Options");
		Utils::Logger::LogInfo(codegenOptions.ToString(), 1);

		Utils::Logger::LogInfo("Target Options");
		Utils::Logger::LogInfo(targetOptions.ToString(), 1);
	}

	Codegen::CodeGenerator<PTX::Bits::Bits64> codegen(codegenOptions, targetOptions);

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

			HorseIR::Analysis::KernelOptionsAnalysis optionsAnalysis(localShapeAnalysis);
			optionsAnalysis.Analyze(function);

			codegen.SetInputOptions(function, optionsAnalysis.GetInputOptions());
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

void FrontendCompiler::Optimize(PTX::Program *program) const
{
}

}
}
