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

	ParameterSpaceAllocator(unsigned int computeCapability) : m_computeCapability(computeCapability) {}

	void Analyze(const FunctionDefinition<VoidType> *function);
	const ParameterSpaceAllocation *GetSpaceAllocation() { return m_allocation; }

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const TypedVariableDeclaration<T, S> *declaration);
	void Visit(const _TypedVariableDeclaration *declaration) override;

private:
	std::size_t GetParameterOffset() const;

	unsigned int m_computeCapability = 0;
	ParameterSpaceAllocation *m_allocation = nullptr;
};

}
}
