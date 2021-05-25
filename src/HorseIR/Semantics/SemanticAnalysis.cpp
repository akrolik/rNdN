#include "HorseIR/Semantics/SemanticAnalysis.h"

#include "HorseIR/Modules/BuiltinModule.h"
#include "HorseIR/Modules/GPUModule.h"

#include "HorseIR/Semantics/DefinitelyAssigned.h"
#include "HorseIR/Semantics/EntryAnalysis.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Semantics/SymbolTable/SymbolTableBuilder.h"
#include "HorseIR/Semantics/TypeChecker.h"

#include "Utils/Chrono.h"

namespace HorseIR {

void SemanticAnalysis::Analyze(Program *program)
{
	auto timeSemantics_start = Utils::Chrono::Start("Semantic analysis");

	// Check the semantic validity of the program

	// Connect the builtin and GPU modules to the program

	program->AddModule(BuiltinModule);
	program->AddModule(GPUModule);

	// Construct the symbol table

	SymbolTableBuilder symbolTable;
	symbolTable.Build(program);

	// Run the type checker

	TypeChecker typeChecker;
	typeChecker.Analyze(program);

	// Check all variables are definitely assigned

	DefinitelyAssigned defAssigned;
	defAssigned.Analyze(program);

	Utils::Chrono::End(timeSemantics_start);
}

const Function *SemanticAnalysis::GetEntry(const Program *program)
{
	// Find the entry point for the program

	EntryAnalysis entryAnalysis;
	entryAnalysis.Analyze(program);
	return entryAnalysis.GetEntry();
}

}
