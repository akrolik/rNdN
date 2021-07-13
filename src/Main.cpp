#include "HorseIR/Optimizer/Optimizer.h"
#include "HorseIR/Semantics/SemanticAnalysis.h"
#include "HorseIR/Transformation/Outliner/Outliner.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Runtime/Interpreter.h"
#include "Runtime/Runtime.h"
#include "Runtime/GPU/Assembler.h"
#include "Runtime/GPU/Compiler.h"

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

	if (!Utils::Options::HasInputFile())
	{
		Utils::Logger::LogError("Missing filename (./r3d3 [options] filename), see --help");
	}
	auto filename = Utils::Options::GetInputFile();

	yyin = fopen(filename.c_str(), "r");
	if (yyin == nullptr)
	{
		Utils::Logger::LogError("Could not open '" + filename + "'");
	}

	Utils::Chrono::End(timeFile_start);

	auto timeCompilation_start = Utils::Chrono::Start("Compilation");
	auto timeSyntax_start = Utils::Chrono::Start("Syntax");

	// Parse the input HorseIR program from stdin and generate an AST

	auto timeParse_start = Utils::Chrono::Start("Parse");

	yyparse();

	Utils::Chrono::End(timeParse_start);

	if (Utils::Options::IsFrontend_PrintHorseIR())
	{
		// Pretty print the input HorseIR program

		Utils::Logger::LogInfo("HorseIR program");

		auto programString = HorseIR::PrettyPrinter::PrettyString(program);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);
	}

	// Collect the symbol table, and verify the program is semantically correct (types+defs)

	HorseIR::SemanticAnalysis::Analyze(program);

	if (Utils::Options::IsFrontend_PrintHorseIRTyped())
	{
		// Pretty print the input HorseIR program

		Utils::Logger::LogInfo("Typed HorseIR program");

		auto programString = HorseIR::PrettyPrinter::PrettyString(program);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Chrono::End(timeSyntax_start);

	if (Utils::Options::IsOptimize_HorseIR())
	{
		// Execute the fixed point optimizer

		HorseIR::Optimizer::Optimizer optimizer;
		optimizer.Optimize(program);
	}

	// Outliner

	HorseIR::Transformation::Outliner outliner;
	auto outlinedProgram = outliner.Outline(program);

	// Compile the program

	Runtime::GPU::Compiler compiler(gpu);
	auto ptxProgram = compiler.Compile(outlinedProgram);

	Runtime::GPU::Assembler assembler(gpu);
	auto gpuProgram = assembler.Assemble(ptxProgram);

	// Load into the GPU manager

	gpu.SetProgram(gpuProgram);

	Utils::Chrono::End(timeCompilation_start);

	if (Utils::Options::IsDebug_CompileOnly())
	{
		Utils::Chrono::Complete();
		std::exit(EXIT_SUCCESS);
	}

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
