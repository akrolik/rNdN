#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include <unordered_map>

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class VirtualRegisterAllocator : public ConstHierarchicalVisitor
{
public:
	void Analyze(const FunctionDefinition<VoidType> *program);
	const RegisterAllocation *GetRegisterAllocation() { return m_allocation; }

	// Functions

	bool VisitIn(const FunctionDefinition<VoidType> *function) override;

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;
	template<class T, class S>
	bool VisitIn(const TypedVariableDeclaration<T, S> *declaration);

private:
	std::uint8_t m_registerOffset = 0;
	std::uint8_t m_predicateOffset = 0;
	RegisterAllocation *m_allocation = nullptr;
};

}
}
