#pragma once

#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"

namespace HorseIR {

class SymbolTableBuilder
{
public:
	void Build(Program *program);
};

class SymbolPass_Modules : public HierarchicalVisitor
{
public:
	using HierarchicalVisitor::VisitIn;
	using HierarchicalVisitor::VisitOut;

	void Build(Program *program);

	bool VisitIn(Program *program) override;
	void VisitOut(Program *program) override;

	bool VisitIn(Module *module) override;
	void VisitOut(Module *module) override;

	// Stop the traversal
	bool VisitIn(GlobalDeclaration *global) override;
	bool VisitIn(FunctionDeclaration *function) override;

private:
	SymbolTable *m_currentSymbolTable = nullptr;
};

class SymbolPass_Imports : public HierarchicalVisitor
{
public:
	using HierarchicalVisitor::VisitIn;
	using HierarchicalVisitor::VisitOut;

	void Build(Program *program);

	bool VisitIn(Program *program) override;
	void VisitOut(Program *program) override;
	
	bool VisitIn(Module *module) override;
	void VisitOut(Module *module) override;

	bool VisitIn(ImportDirective *import) override;

	// Stop the traversal
	bool VisitIn(ModuleContent *content) override;

private:
	SymbolTable *m_globalSymbolTable = nullptr;
	SymbolTable *m_currentImportTable = nullptr;
};

class SymbolPass_Functions : public HierarchicalVisitor
{
public:
	using HierarchicalVisitor::VisitIn;
	using HierarchicalVisitor::VisitOut;

	void Build(Program *program);

	bool VisitIn(Program *program) override;
	void VisitOut(Program *program) override;

	bool VisitIn(Module *module) override;
	void VisitOut(Module *module) override;

	bool VisitIn(Function *function) override;
	void VisitOut(Function *function) override;

	bool VisitIn(BlockStatement *blockS) override;
	void VisitOut(BlockStatement *blockS) override;

	bool VisitIn(AssignStatement *assignS) override;

	bool VisitIn(VariableDeclaration *declaration) override;
	bool VisitIn(Identifier *identifier) override;

	void VisitOut(FunctionLiteral *literal) override;

private:
	SymbolTable::Symbol *LookupIdentifier(const Identifier *identifier);

	SymbolTable *m_currentSymbolTable = nullptr;
};

}
