#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/SpaceAllocator/GlobalSpaceAllocation.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class GlobalSpaceAllocator : public ConstHierarchicalVisitor, public ConstDeclarationVisitor
{
public:
	void Analyze(const Module *module);
	const GlobalSpaceAllocation *GetSpaceAllocation() { return m_allocation; }

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const TypedVariableDeclaration<T, S> *declaration);
	void Visit(const _TypedVariableDeclaration *declaration) override;

	// Functions

	bool VisitIn(const FunctionDefinition<VoidType> *function) override;

private:
	GlobalSpaceAllocation *m_allocation = nullptr;
};

}
}
