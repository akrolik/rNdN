#pragma once

#include "Backend/Codegen/Generators/Generator.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"
#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

#include "PTX/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class ControlFlowGenerator : public Generator, public PTX::ConstHierarchicalVisitor, public PTX::ConstDeclarationVisitor, public PTX::Analysis::ConstStructuredGraphVisitor
{
public:
	using Generator::Generator;

	// Public API

	virtual void Generate(const PTX::FunctionDefinition<PTX::VoidType> *function);

	// Declarations

	bool VisitIn(const PTX::VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const PTX::TypedVariableDeclaration<T, S> *declaration);
	void Visit(const PTX::_TypedVariableDeclaration *declaration) override;

	// Basic Block
	
	bool VisitIn(const PTX::BasicBlock *block) override;
	void VisitOut(const PTX::BasicBlock *block) override;

	// Statements

	bool VisitIn(const PTX::InstructionStatement *statement) override;
};

}
}
