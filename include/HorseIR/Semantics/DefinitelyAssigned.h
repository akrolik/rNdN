#pragma once

#include <unordered_set>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"

namespace HorseIR {

class DefinitelyAssigned : public ConstHierarchicalVisitor
{
public:
	using ConstHierarchicalVisitor::VisitIn;
	using ConstHierarchicalVisitor::VisitOut;

	void Analyze(const Program *program);

	//TODO: Need to traverse globals first!

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
	std::unordered_set<const SymbolTable::Symbol *> m_definitions;
};

}
