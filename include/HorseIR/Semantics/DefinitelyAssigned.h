#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"

#include "Libraries/robin_hood.h"

namespace HorseIR {

class DefinitelyAssigned : public ConstHierarchicalVisitor
{
public:
	using ConstHierarchicalVisitor::VisitIn;
	using ConstHierarchicalVisitor::VisitOut;

	void Analyze(const Program *program);

	bool VisitIn(const GlobalDeclaration *global) override;
	bool VisitIn(const Function *function) override;
	bool VisitIn(const Parameter *parameter) override;
	
	// Statements

	bool VisitIn(const AssignStatement *assignS) override;
	bool VisitIn(const IfStatement *assignS) override;
	bool VisitIn(const WhileStatement *whileS) override;
	bool VisitIn(const RepeatStatement *repeatS) override;

	// Expressions

	bool VisitIn(const FunctionLiteral *literal) override;
	bool VisitIn(const Identifier *identifier) override;

private:
	bool m_globalsPhase = false;

	robin_hood::unordered_set<const SymbolTable::Symbol *> m_globals;
	robin_hood::unordered_set<const SymbolTable::Symbol *> m_definitions;
};

}
