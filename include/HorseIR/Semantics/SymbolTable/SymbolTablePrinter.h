#pragma once

#include <string>
#include <sstream>

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"

namespace HorseIR {

class SymbolTablePrinter : public ConstHierarchicalVisitor
{
public:
	using ConstHierarchicalVisitor::VisitIn;
	using ConstHierarchicalVisitor::VisitOut;

	static std::string PrettyString(const Program *program);

	bool VisitIn(const Program *program) override;
	void VisitOut(const Program *program) override;

	bool VisitIn(const Module *module) override;
	void VisitOut(const Module *module) override;

	bool VisitIn(const Function *function) override;
	void VisitOut(const Function *function) override;

	bool VisitIn(const BlockStatement *blockS) override;
	void VisitOut(const BlockStatement *blockS) override;

	bool VisitIn(const VariableDeclaration *declaration) override;

private:
	void Indent();
	unsigned int m_indent = 0;
	std::stringstream m_string;

	const SymbolTable *m_currentSymbolTable = nullptr;
};

}
