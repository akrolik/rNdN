#include "HorseIR/Analysis/FlowAnalysisPrinter.h"
#include "HorseIR/Modules/BuiltinModule.h"
#include "HorseIR/Semantics/DefinitelyAssigned.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTableBuilder.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTablePrinter.h"
#include "HorseIR/Semantics/TypeChecker.h"
#include "HorseIR/Tree/Program.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Analysis/EntryAnalysis.h"
#include "Analysis/BasicFlow/ReachingDefinitions.h"
#include "Analysis/BasicFlow/LiveVariables.h"
#include "Analysis/BasicFlow/UDDUChainsBuilder.h"

#include "Transformation/Outliner/Outliner.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

// #include "Interpreter/Interpreter.h"

// #include "Runtime/Runtime.h"

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

	// Check the semantic validity of the program

	Utils::Logger::LogSection("Analyzing input program");

	auto timeSymbols_start = Utils::Chrono::Start();

	// Connect the builtin module to the program

	program->AddModule(HorseIR::BuiltinModule);

	// Construct the symbol table

	HorseIR::SymbolTableBuilder symbolTable;
	symbolTable.Build(program);

	auto timeSymbols = Utils::Chrono::End(timeSymbols_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_symbol))
	{
		// Pretty print the symbol table cactus stack

		Utils::Logger::LogInfo("HorseIR symbol table");

		auto tableString = HorseIR::SymbolTablePrinter::PrettyString(program);
		Utils::Logger::LogInfo(tableString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Symbol table", timeSymbols);

	// Run the type checker
	
	auto timeTypes_start = Utils::Chrono::Start();

	HorseIR::TypeChecker typeChecker;
	typeChecker.Analyze(program);

	auto timeTypes = Utils::Chrono::End(timeTypes_start);
	Utils::Logger::LogTiming("Typechecker", timeTypes);

	// Check all variables are definitely assigned
	
	auto timeAssigned_start = Utils::Chrono::Start();

	HorseIR::DefinitelyAssigned defAssigned;
	defAssigned.Analyze(program);

	auto timeAssigned = Utils::Chrono::End(timeAssigned_start);
	Utils::Logger::LogTiming("Definitely asigned", timeAssigned);

	// Find the entry point for the program

	auto timeEntry_start = Utils::Chrono::Start();

	Analysis::EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	auto entry = entryAnalysis.GetEntry();

	auto timeEntry = Utils::Chrono::End(timeEntry_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Found entry point '" + entry->GetName() + "'");
	}

	Utils::Logger::LogTiming("Entry analysis", timeEntry);

	// Reaching definitions

	auto timeReachingDefs_start = Utils::Chrono::Start();

	Analysis::ReachingDefinitions reachingDefs(program);
	reachingDefs.Analyze(entry);

	auto timeReachingDefs = Utils::Chrono::End(timeReachingDefs_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Reaching definitions");

		auto reachingDefsString = HorseIR::FlowAnalysisPrinter<Analysis::ReachingDefinitionsProperties>::PrettyString(reachingDefs, entry);
		Utils::Logger::LogInfo(reachingDefsString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Reaching definitions", timeReachingDefs);

	// UD/DU chain builder
	
	auto timeUDDU_start = Utils::Chrono::Start();

	Analysis::UDDUChainsBuilder useDefs(reachingDefs);
	useDefs.Build(entry);

	auto timeUDDU = Utils::Chrono::End(timeUDDU_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("UD/DU chains");

		auto useDefsString = useDefs.DebugString();
		Utils::Logger::LogInfo(useDefsString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("UD/DU chains", timeUDDU);

	// Live variables

	auto timeLiveVariables_start = Utils::Chrono::Start();

	Analysis::LiveVariables liveVariables(program);
	liveVariables.Analyze(entry);

	auto timeLiveVariables = Utils::Chrono::End(timeLiveVariables_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Live variables");

		auto liveVariablesString = HorseIR::FlowAnalysisPrinter<Analysis::LiveVariablesProperties>::PrettyString(liveVariables, entry);
		Utils::Logger::LogInfo(liveVariablesString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Live variables", timeLiveVariables);

	// Outliner

	Transformation::Outliner outliner(program);
	outliner.Outline(entry);

	// Execute the HorseIR entry method in an interpeter, compiling GPU sections as needed

	//TODO: Update runtime for new structure
	/*
	Utils::Logger::LogSection("Starting program execution");

	// Initialize the runtime environment and check that the machine is capable of running the query

	Runtime::Runtime runtime;
	runtime.Initialize();

	Interpreter::Interpreter interpreter(runtime);
	auto result = interpreter.Execute(entry, {});
	result->Dump();
	*/
}
