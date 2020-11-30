#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include <unordered_map>

#include "PTX/PTX.h"
#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"

namespace PTX {
namespace Analysis {

class VirtualRegisterAllocator : public ConstHierarchicalVisitor
{
public:
	void Analyze(const Program *program);
	const std::unordered_map<const FunctionDefinition<VoidType> *, const RegisterAllocation *>& GetRegisterAllocations() { return m_allocations; }

	// Functions

	bool VisitIn(const FunctionDefinition<VoidType> *function) override;
	void VisitOut(const FunctionDefinition<VoidType> *function) override;

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;
	template<class T, class S>
	bool VisitIn(const TypedVariableDeclaration<T, S> *declaration);

private:
	std::uint8_t m_registerOffset = 0;
	std::uint8_t m_predicateOffset = 0;
	RegisterAllocation *m_currentAllocation = nullptr;

	std::unordered_map<const FunctionDefinition<VoidType> *, const RegisterAllocation *> m_allocations;
};

}
}
