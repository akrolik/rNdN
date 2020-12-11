#include "HorseIR/Semantics/SemanticAnalysis.h"

#include "HorseIR/Modules/BuiltinModule.h"
#include "HorseIR/Modules/GPUModule.h"

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
	auto timeSemantics_start = Utils::Chrono::Start("Semantic analysis");

	// Check the semantic validity of the program

	auto timeSymbols_start = Utils::Chrono::Start("Symbol table");

	// Connect the builtin and GPU modules to the program

	program->AddModule(BuiltinModule);
	program->AddModule(GPUModule);

	// Construct the symbol table

	SymbolTableBuilder symbolTable;
	symbolTable.Build(program);

	Utils::Chrono::End(timeSymbols_start);

	if (Utils::Options::IsFrontend_PrintSymbols())
	{
		// Pretty print the symbol table cactus stack

		Utils::Logger::LogInfo("HorseIR symbol table");

		auto tableString = SymbolTablePrinter::PrettyString(program);
		Utils::Logger::LogInfo(tableString, 0, true, Utils::Logger::NoPrefix);
	}

	// Run the type checker

	auto timeTypes_start = Utils::Chrono::Start("Typechecker");

	TypeChecker typeChecker;
	typeChecker.Analyze(program);

	Utils::Chrono::End(timeTypes_start);

	// Check all variables are definitely assigned

	auto timeAssigned_start = Utils::Chrono::Start("Definitely assigned");

	DefinitelyAssigned defAssigned;
	defAssigned.Analyze(program);

	Utils::Chrono::End(timeAssigned_start);
	Utils::Chrono::End(timeSemantics_start);
}

const Function *SemanticAnalysis::GetEntry(const Program *program)
{
	// Find the entry point for the program

	auto timeEntry_start = Utils::Chrono::Start("Entry analysis");

	EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	auto entry = entryAnalysis.GetEntry();

	Utils::Chrono::End(timeEntry_start);

	if (Utils::Options::IsFrontend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Found entry point '" + entry->GetName() + "'");
	}

	return entry;
}

}
