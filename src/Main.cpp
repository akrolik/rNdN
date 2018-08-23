#include "HorseIR/BuiltinModule.h"
#include "HorseIR/Analysis/EntryAnalysis.h"
#include "HorseIR/Analysis/ShapeAnalysis.h"
#include "HorseIR/Analysis/SymbolTable.h"
#include "HorseIR/Analysis/TypeAnalysis.h"
#include "HorseIR/Tree/Program.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#include "Interpreter/Interpreter.h"

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

	// Initialize the runtime environment and check that the machine is capable of running the query

	Runtime::Runtime runtime;
	runtime.Initialize();

	// Parse the input HorseIR program from stdin and generate an AST

	Utils::Logger::LogSection("Parsing input program");
	auto timeFrontend_start = Utils::Chrono::Start();

	yyparse();
	
	auto timeFrontend = Utils::Chrono::End(timeFrontend_start);

	if (Utils::Options::Present(Utils::Options::Opt_Dump_hir))
	{
		// Pretty print the input HorseIR program

		Utils::Logger::LogInfo("Input HorseIR program");
		Utils::Logger::LogInfo(program->ToString(), 0, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("HorseIR frontend", timeFrontend);

	Utils::Logger::LogSection("Analyzing input program");
	auto timeSymbols_start = Utils::Chrono::Start();

	// Connect the builtin module to the program

	program->AddModule(HorseIR::BuiltinModule);

	// Construct the symbol table

	HorseIR::SymbolTableBuilder symbolTable;
	symbolTable.Build(program);

	auto timeSymbols = Utils::Chrono::End(timeSymbols_start);

	if (Utils::Options::Present(Utils::Options::Opt_Dump_symbol))
	{
		// Dump the symbol table to stdout

		HorseIR::SymbolTableDumper dump;
		dump.Dump(program);
	}
	Utils::Logger::LogTiming("Symbol table", timeSymbols);

	// Run the type checker

	auto timeTypes_start = Utils::Chrono::Start();

	HorseIR::TypeAnalysis typeAnalysis;
	typeAnalysis.Analyze(program);

	auto timeTypes = Utils::Chrono::End(timeTypes_start);

	if (Utils::Options::Present(Utils::Options::Opt_Dump_type))
	{
		// Dump the type checking results to stdout

		// HorseIR::TypeAnalysisDumper dump;
		// dump.Dump(program);
	}
	Utils::Logger::LogTiming("Typechecker", timeTypes);

	// Find the entry point for the program

	auto timeEntry_start = Utils::Chrono::Start();

	HorseIR::EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	auto entry = entryAnalysis.GetEntry();

	auto timeEntry = Utils::Chrono::End(timeEntry_start);

	Utils::Logger::LogInfo("Found entry point '" + entry->GetName() + "'");
	Utils::Logger::LogTiming("Entry analysis", timeEntry);

	// Perform a conservative shape analysis

	auto timeShapes_start = Utils::Chrono::Start();

	HorseIR::ShapeAnalysis shapeAnalysis;
	shapeAnalysis.Analyze(entry);

	auto timeShapes = Utils::Chrono::End(timeShapes_start);

	if (Utils::Options::Present(Utils::Options::Opt_Dump_shape))
	{
		// Dump the shape analysis results to stdout

		// HorseIR::ShapeAnalysisDumper dump;
		// dump.Dump(program);
	}
	Utils::Logger::LogTiming("Shape analysis", timeShapes);

	// Execute the HorseIR entry method in an interpeter, compiling GPU sections as needed

	Utils::Logger::LogSection("Starting program execution");

	Interpreter::Interpreter interpreter(runtime);
	auto result = interpreter.Execute(entry, {});
	result->Dump();
}
