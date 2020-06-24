#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Semantics/SemanticAnalysis.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Optimizer/Optimizer.h"

#include "Runtime/Interpreter.h"
#include "Runtime/Runtime.h"
#include "Runtime/GPU/GPUAssembler.h"
#include "Runtime/GPU/GPUCompiler.h"

#include "Transformation/Outliner/Outliner.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

extern FILE *yyin;
int yyparse();

HorseIR::Program *program;

int main(int argc, const char *argv[])
{
	// Initialize the input arguments from the command line

	Utils::Options::Initialize(argc, argv);

	// Initialize the runtime environment and check that the machine is capable of running the query

	Utils::Chrono::Initialize();

	auto& runtime = *Runtime::Runtime::GetInstance();
	runtime.Initialize();

	auto& gpu = runtime.GetGPUManager();

	// Parse the input HorseIR program from stdin and generate an AST

	Utils::Logger::LogSection("Parsing HorseIR program");

	auto timeFile_start = Utils::Chrono::Start("File open");

	if (!Utils::Options::Present(Utils::Options::Opt_File))
	{
		Utils::Logger::LogError("Missing filename (./r3d3 [options] filename), see --help");
	}
	auto filename = Utils::Options::Get<std::string>(Utils::Options::Opt_File);

	yyin = fopen(filename.c_str(), "r");
	if (yyin == nullptr)
	{
		Utils::Logger::LogError("Could not open '" + filename + "'");
	}

	Utils::Chrono::End(timeFile_start);

	auto timeCompilation_start = Utils::Chrono::Start("Compilation");
	auto timeFrontend_start = Utils::Chrono::Start("Frontend");

	// Parse the input HorseIR program from stdin and generate an AST

	auto timeParse_start = Utils::Chrono::Start("Parse");

	yyparse();

	Utils::Chrono::End(timeParse_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_hir))
	{
		// Pretty print the input HorseIR program

		Utils::Logger::LogInfo("HorseIR program");

		auto programString = HorseIR::PrettyPrinter::PrettyString(program);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);
	}

	// Collect the symbol table, and verify the program is semantically correct (types+defs)

	HorseIR::SemanticAnalysis::Analyze(program);

	if (Utils::Options::Present(Utils::Options::Opt_Print_hir_typed))
	{
		// Pretty print the input HorseIR program

		Utils::Logger::LogInfo("Typed HorseIR program");

		auto programString = HorseIR::PrettyPrinter::PrettyString(program);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Chrono::End(timeFrontend_start);

	if (Utils::Options::Present(Utils::Options::Opt_Optimize))
	{
		// Execute the fixed point optimizer

		Optimizer::Optimizer optimizer;
		optimizer.Optimize(program);
	}

	// Outliner

	Transformation::Outliner outliner;
	outliner.Outline(program);

	auto outlinedProgram = outliner.GetOutlinedProgram();

	// Compile the program

	Runtime::GPUCompiler compiler(gpu);
	auto ptxProgram = compiler.Compile(outlinedProgram);

	Runtime::GPUAssembler assembler(gpu);
	auto gpuProgram = assembler.Assemble(ptxProgram);

	// Load into the GPU manager

	gpu.SetProgram(gpuProgram);

	Utils::Chrono::End(timeCompilation_start);

	// Load data

	Utils::Logger::LogSection("Loading data");

	runtime.LoadData();

	// Execute the HorseIR entry function in an interpeter, compiling GPU sections as needed

	Utils::Logger::LogSection("Executing program");

	auto timeExecution_start = Utils::Chrono::Start("Execution");

	Runtime::Interpreter interpreter(runtime);
	auto results = interpreter.Execute(outlinedProgram);

	auto timeOutput_start = Utils::Chrono::Start("Output collection");

	for (const auto& result : results)
	{
		result->SetTag("output");
		result->ValidateCPU();
	}

	Utils::Chrono::End(timeOutput_start);
	Utils::Chrono::End(timeExecution_start);

	// Print the results

	Utils::Logger::LogSection("Execution result");

	for (const auto& result : results)
	{
		Utils::Logger::LogInfo(result->DebugDump(), 0, true, Utils::Logger::NoPrefix);
	}

	// Runtime::Runtime::Destroy();

	Utils::Chrono::Complete();
}
