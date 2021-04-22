#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/BasicFlow/LiveIntervals.h"
#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

class LinearScanRegisterAllocator : public ConstHierarchicalVisitor, public ConstDeclarationVisitor
{
public:
	LinearScanRegisterAllocator(const LiveIntervals& liveIntervals) : m_liveIntervals(liveIntervals) {}

	// Public API

	void Analyze(const FunctionDefinition<VoidType> *function);
	const RegisterAllocation *GetRegisterAllocation() { return m_allocation; }

	// Functions

	void VisitOut(const FunctionDefinition<VoidType> *function) override;

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const TypedVariableDeclaration<T, S> *declaration);
	void Visit(const _TypedVariableDeclaration *declaration) override;

private:
	const LiveIntervals& m_liveIntervals;

	RegisterAllocation *m_allocation = nullptr;
	robin_hood::unordered_map<std::string, Bits> m_registerBits;
};

}
}
