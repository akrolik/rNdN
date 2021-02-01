#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "Backend/Codegen/Builder.h"

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Analysis/SpaceAllocator/GlobalSpaceAllocation.h"
#include "PTX/Analysis/SpaceAllocator/LocalSpaceAllocation.h"
#include "PTX/Tree/Tree.h"

#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class CodeGenerator : public PTX::ConstHierarchicalVisitor, public PTX::ConstDeclarationVisitor
{
public:
	CodeGenerator(const PTX::Analysis::GlobalSpaceAllocation *globalSpaceAllocation) : m_builder(globalSpaceAllocation) {}

	SASS::Function *Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *registerAllocation, const PTX::Analysis::LocalSpaceAllocation *spaceAllocation);

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

private:
	Builder m_builder;
};

}
}
