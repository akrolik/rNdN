#pragma once

#include <stack>
#include <unordered_set>
#include <vector>

#include "Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Tree/Tree.h"

namespace Transformation {

class OutlineBuilder : public Analysis::DependencyOverlayConstVisitor, public HorseIR::ConstVisitor
{
public:
	// Transformation input and output

	void Build(const Analysis::FunctionDependencyOverlay *overlay);

	const std::vector<HorseIR::Function *>& GetFunctions() const { return m_functions; }

	void Visit(const HorseIR::Statement *statement) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;

	// Dependency overlay visitors

	void Visit(const Analysis::DependencyOverlay *overlay) override;

	void Visit(const Analysis::FunctionDependencyOverlay *overlay) override;
	void Visit(const Analysis::IfDependencyOverlay *overlay) override;
	void Visit(const Analysis::WhileDependencyOverlay *overlay) override;
	void Visit(const Analysis::RepeatDependencyOverlay *overlay) override;

private:
	std::vector<HorseIR::Function *> m_functions;
	std::stack<std::vector<HorseIR::Statement *>> m_statements;
	std::stack<std::unordered_set<const HorseIR::SymbolTable::Symbol *>> m_symbols;

	static const HorseIR::Type *GetType(const HorseIR::SymbolTable::Symbol *symbol);
	void BuildDeclarations();
	void InsertStatement(HorseIR::Statement *statement);
	void InsertDeclaration(HorseIR::DeclarationStatement *declaration);

	unsigned int m_kernelIndex = 1;
	const HorseIR::Function *m_currentFunction = nullptr;
};

}
