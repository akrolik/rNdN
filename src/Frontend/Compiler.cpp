#include "Frontend/Compiler.h"

#include "HorseIR/Analysis/DataObject/DataObjectAnalysis.h"
#include "HorseIR/Analysis/Geometry/KernelOptionsAnalysis.h"
#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"
#include "HorseIR/Analysis/Shape/ShapeAnalysis.h"

#include "Frontend/Codegen/CodeGenerator.h"
#include "Frontend/Codegen/InputOptions.h"
#include "Frontend/Codegen/TargetOptions.h"

#include "HorseIR/Semantics/SemanticAnalysis.h"

#include "PTX/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"
#include "Utils/Options.h"

namespace Frontend {

PTX::Program *Compiler::Compile(const HorseIR::Program *program) const
{
	auto timeCompiler_start = Utils::Chrono::Start("Frontend compiler");

	// Get the entry function

	auto entry = HorseIR::SemanticAnalysis::GetEntry(program);

	// Collect input/output shapes and data objects interprocedurally

	HorseIR::Analysis::DataObjectAnalysis dataAnalysis(program);
	dataAnalysis.SetCollectInSets(false);
	dataAnalysis.SetCollectOutSets(false);
	dataAnalysis.Analyze(entry);

	HorseIR::Analysis::ShapeAnalysis shapeAnalysis(dataAnalysis, program);
	shapeAnalysis.SetCollectOutSets(false);
	shapeAnalysis.Analyze(entry);

	// Generate 64-bit PTX code from the input HorseIR for the current device

	if (Utils::Options::IsFrontend_PrintPTX())
	{
		Utils::Logger::LogSection("Generating PTX program");
	}

	// Collect the device properties for codegen

	Codegen::TargetOptions targetOptions;
	targetOptions.ComputeCapability = m_device->GetComputeCapability();
	targetOptions.MaxBlockSize = m_device->GetMaxThreadsDimension(0);
	targetOptions.WarpSize = m_device->GetWarpSize();
	targetOptions.SharedMemorySize = m_device->GetSharedMemorySize();

	// Initialize the geometry information for code generation

	if (Utils::Options::IsFrontend_PrintPTX())
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

			HorseIR::Analysis::KernelOptionsAnalysis optionsAnalysis(localShapeAnalysis);
			auto options = optionsAnalysis.Analyze(function);

			codegen.SetInputOptions(function, options);
		}
	}

	// Generate the program

	auto timePTX_start = Utils::Chrono::Start("PTX codegen");

	auto ptxProgram = codegen.Generate(program);

	Utils::Chrono::End(timePTX_start);
	Utils::Chrono::End(timeCompiler_start);

	// Dump the PTX program or JSON string to stdout

	if (Utils::Options::IsFrontend_PrintPTX())
	{
		Utils::Logger::LogInfo("Generated PTX program");

		auto programString = PTX::PrettyPrinter::PrettyString(ptxProgram);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);

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

	if (Utils::Options::IsFrontend_PrintJSON())
	{
		Utils::Logger::LogInfo("Generated PTX program (JSON)");
		Utils::Logger::LogInfo(ptxProgram->ToJSON().dump(4), 0, true, Utils::Logger::NoPrefix);
	}

	// Optimize the generated PTX program

	if (Utils::Options::IsOptimize_PTX())
	{
		Optimize(ptxProgram);
	}

	return ptxProgram;
}

void Compiler::Optimize(PTX::Program *program) const
{
}

}
