#include "HorseIR/BuiltinModule.h"
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

	Utils::Logger::LogSection("Building symbol table");
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

	auto timeTypes_start = Utils::Chrono::Start();

	// Run the type checker

	HorseIR::TypeAnalysis typeAnalysis;
	typeAnalysis.Analyze(program);

	auto timeTypes = Utils::Chrono::End(timeTypes_start);

	if (Utils::Options::Present(Utils::Options::Opt_Dump_type))
	{
		// Dump the type checking results to stdout

		//TODO: Type dumper
	}
	Utils::Logger::LogTiming("Type checker", timeTypes);

	// Execute the HorseIR program in an interpeter, compiling GPU sections as needed

	Interpreter::Interpreter interpreter(runtime);
	interpreter.Execute(program);
}
