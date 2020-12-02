#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"

#include "SASS/SASS.h"

namespace Backend {

class CodeGenerator : public PTX::ConstHierarchicalVisitor
{
public:
	SASS::Function *Generate(const PTX::FunctionDefinition<PTX::VoidType> *function, const PTX::Analysis::RegisterAllocation *allocation);

	// Functions

	bool VisitIn(const PTX::FunctionDefinition<PTX::VoidType> *function) override;
	void VisitOut(const PTX::FunctionDefinition<PTX::VoidType> *function) override;

	// Declarations

	bool VisitIn(const PTX::VariableDeclaration *declaration) override;
	template<class T, class S>
	bool VisitIn(const PTX::TypedVariableDeclaration<T, S> *declaration);

private:
	SASS::Function *m_function = nullptr;
	const PTX::Analysis::RegisterAllocation *m_allocation = nullptr;
};

}
