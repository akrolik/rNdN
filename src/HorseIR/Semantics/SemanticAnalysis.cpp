#include "HorseIR/Semantics/SemanticAnalysis.h"

#include "HorseIR/Modules/BuiltinModule.h"

#include "HorseIR/Semantics/DefinitelyAssigned.h"
#include "HorseIR/Semantics/EntryAnalysis.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTableBuilder.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTablePrinter.h"
#include "HorseIR/Semantics/TypeChecker.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {

void SemanticAnalysis::Analyze(Program *program)
{
	// Check the semantic validity of the program

	Utils::Logger::LogSection("Analyzing input program");

	auto timeSymbols_start = Utils::Chrono::Start();

	// Connect the builtin module to the program

	program->AddModule(BuiltinModule);

	// Construct the symbol table

	SymbolTableBuilder symbolTable;
	symbolTable.Build(program);

	auto timeSymbols = Utils::Chrono::End(timeSymbols_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_symbol))
	{
		// Pretty print the symbol table cactus stack

		Utils::Logger::LogInfo("HorseIR symbol table");

		auto tableString = SymbolTablePrinter::PrettyString(program);
		Utils::Logger::LogInfo(tableString, 0, true, Utils::Logger::NoPrefix);
	}

	Utils::Logger::LogTiming("Symbol table", timeSymbols);

	// Run the type checker

	auto timeTypes_start = Utils::Chrono::Start();

	TypeChecker typeChecker;
	typeChecker.Analyze(program);

	auto timeTypes = Utils::Chrono::End(timeTypes_start);
	Utils::Logger::LogTiming("Typechecker", timeTypes);

	// Check all variables are definitely assigned

	auto timeAssigned_start = Utils::Chrono::Start();

	DefinitelyAssigned defAssigned;
	defAssigned.Analyze(program);

	auto timeAssigned = Utils::Chrono::End(timeAssigned_start);
	Utils::Logger::LogTiming("Definitely asigned", timeAssigned);
}

Function *SemanticAnalysis::GetEntry(Program *program)
{
	// Find the entry point for the program

	auto timeEntry_start = Utils::Chrono::Start();

	EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	auto entry = entryAnalysis.GetEntry();

	auto timeEntry = Utils::Chrono::End(timeEntry_start);

	if (Utils::Options::Present(Utils::Options::Opt_Print_analysis))
	{
		Utils::Logger::LogInfo("Found entry point '" + entry->GetName() + "'");
	}

	Utils::Logger::LogTiming("Entry analysis", timeEntry);

	return entry;
}

}
