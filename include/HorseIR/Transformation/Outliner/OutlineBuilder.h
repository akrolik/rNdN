#pragma once

#include <stack>
#include <vector>

#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Tree/Tree.h"

#include "HorseIR/Transformation/Outliner/OutlineLibrary.h"

#include "Libraries/robin_hood.h"

namespace HorseIR {
namespace Transformation {

class OutlineBuilder : public Analysis::DependencyOverlayConstVisitor, public ConstVisitor
{
public:
	OutlineBuilder() : m_libraryOutliner(m_functions, m_symbols) {}

	// Transformation input and output

	const std::vector<Function *>& Build(const Analysis::FunctionDependencyOverlay *overlay);

	void Visit(const Statement *statement) override;
	void Visit(const AssignStatement *assignS) override;

	// Dependency overlay visitors

	void Visit(const Analysis::DependencyOverlay *overlay) override;

	void Visit(const Analysis::FunctionDependencyOverlay *overlay) override;
	void Visit(const Analysis::IfDependencyOverlay *overlay) override;
	void Visit(const Analysis::WhileDependencyOverlay *overlay) override;
	void Visit(const Analysis::RepeatDependencyOverlay *overlay) override;

private:
	std::vector<Function *> m_functions;
	std::stack<std::vector<Statement *>> m_statements;
	std::stack<robin_hood::unordered_set<const SymbolTable::Symbol *>> m_symbols;

	OutlineLibrary m_libraryOutliner;

	static const Type *GetType(const SymbolTable::Symbol *symbol);
	void BuildDeclarations();
	void InsertStatement(Statement *statement);
	void InsertDeclaration(DeclarationStatement *declaration);

	unsigned int m_kernelIndex = 1;
	const Function *m_currentFunction = nullptr;
};

}
}
