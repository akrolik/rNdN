#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "HorseIR/Semantics/SemanticAnalysis.h"
#include "Optimizer/Optimizer.h"
#include "Transformation/Outliner/Outliner.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#include "Runtime/Interpreter.h"
#include "Runtime/Runtime.h"

//TODO: Move these somewhere
// #include "PTX/ArithmeticTest.h"
// #include "PTX/ComparisonTest.h"
// #include "PTX/ControlFlowTest.h"
// #include "PTX/DataTest.h"
// #include "PTX/LogicalTest.h"
// #include "PTX/ShiftTest.h"
// #include "PTX/SynchronizationTest.h"

// #include "PTX/AddTest.h"
// #include "PTX/BasicTest.h"
// #include "PTX/ConditionalTest.h"

int yyparse();

HorseIR::Program *program;

int main(int argc, const char *argv[])
{
	// Initialize the input arguments from the command line

	Utils::Options::Initialize(argc, argv);

	// Parse the input HorseIR program from stdin and generate an AST

	Utils::Logger::LogSection("Parsing input program");

	auto timeFrontend_start = Utils::Chrono::Start();

	yyparse();
	
	auto timeFrontend = Utils::Chrono::End(timeFrontend_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_hir))
	{
		// Pretty print the input HorseIR program

		Utils::Logger::LogInfo("HorseIR program");

		auto programString = HorseIR::PrettyPrinter::PrettyString(program);
		Utils::Logger::LogInfo(programString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("HorseIR frontend", timeFrontend);

	// Collect the symbol table, and verify the program is semantically correct (types+defs)

	HorseIR::SemanticAnalysis::Analyze(program);

	// Execute the fixed point optimizer

	Optimizer::Optimizer optimizer;
	optimizer.Optimize(program);

	// Outliner

	Transformation::Outliner outliner;
	outliner.Outline(program);
	auto outlinedProgram = outliner.GetOutlinedProgram();

	// Re-run the semantic analysis to build the AST symbol table links
	
	HorseIR::SemanticAnalysis::Analyze(outlinedProgram);
	auto outlinedEntry = HorseIR::SemanticAnalysis::GetEntry(outlinedProgram);

	// Execute the HorseIR entry function in an interpeter, compiling GPU sections as needed

	Utils::Logger::LogSection("Starting program execution");

	// Initialize the runtime environment and check that the machine is capable of running the query

	Runtime::Runtime runtime;
	runtime.Initialize();

	Runtime::Interpreter interpreter(runtime, outlinedProgram);
	auto results = interpreter.Execute(outlinedEntry, {});

	Utils::Logger::LogSection("Execution result");

	for (const auto& result : results)
	{
		Utils::Logger::LogInfo(result->DebugDump(), 0, true, Utils::Logger::NoPrefix);
	}
}
