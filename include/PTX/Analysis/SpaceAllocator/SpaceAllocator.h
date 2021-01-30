#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/SpaceAllocator/SpaceAllocation.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class SpaceAllocator : public ConstHierarchicalVisitor, public ConstDeclarationVisitor
{
public:
	void Analyze(const FunctionDefinition<VoidType> *function);
	const SpaceAllocation *GetSpaceAllocation() { return m_allocation; }

	// Declarations

	bool VisitIn(const PTX::VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const PTX::TypedVariableDeclaration<T, S> *declaration);
	void Visit(const PTX::_TypedVariableDeclaration *declaration) override;

private:
	SpaceAllocation *m_allocation = nullptr;
};

}
}
