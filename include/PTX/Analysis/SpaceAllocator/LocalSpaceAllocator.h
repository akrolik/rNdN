#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/SpaceAllocator/GlobalSpaceAllocation.h"
#include "PTX/Analysis/SpaceAllocator/LocalSpaceAllocation.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class LocalSpaceAllocator : public ConstHierarchicalVisitor, public ConstDeclarationVisitor
{
public:
	LocalSpaceAllocator(const GlobalSpaceAllocation *globalAllocation) : m_globalAllocation(globalAllocation) {}

	void Analyze(const FunctionDefinition<VoidType> *function);
	const LocalSpaceAllocation *GetSpaceAllocation() { return m_allocation; }

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const TypedVariableDeclaration<T, S> *declaration);
	void Visit(const _TypedVariableDeclaration *declaration) override;

private:
	const GlobalSpaceAllocation *m_globalAllocation = nullptr;
	LocalSpaceAllocation *m_allocation = nullptr;
};

}
}
