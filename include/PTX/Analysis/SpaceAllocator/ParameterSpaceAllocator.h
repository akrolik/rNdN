#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocation.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class ParameterSpaceAllocator : public ConstHierarchicalVisitor, public ConstDeclarationVisitor
{
public:
	inline const static std::string Name = "Parameter space allocator";
	inline const static std::string ShortName = "param";

	void Analyze(const FunctionDefinition<VoidType> *function);
	const ParameterSpaceAllocation *GetSpaceAllocation() { return m_allocation; }

	// Declarations

	bool VisitIn(const FunctionDefinition<VoidType> *function) override;
	bool VisitIn(const VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const TypedVariableDeclaration<T, S> *declaration);
	void Visit(const _TypedVariableDeclaration *declaration) override;

private:
	ParameterSpaceAllocation *m_allocation = nullptr;
};

}
}
