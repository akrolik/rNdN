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
	inline const static std::string Name = "Linear scan allocator";
	inline const static std::string ShortName = "reg";

	LinearScanRegisterAllocator(const LiveIntervals& liveIntervals, unsigned int computeCapability)
		: m_liveIntervals(liveIntervals), m_computeCapability(computeCapability) {}

	// Public API

	void Analyze(const FunctionDefinition<VoidType> *function);
	const RegisterAllocation *GetRegisterAllocation() { return m_allocation; }

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const TypedVariableDeclaration<T, S> *declaration);
	void Visit(const _TypedVariableDeclaration *declaration) override;

private:
	const LiveIntervals& m_liveIntervals;
	unsigned int m_computeCapability = 0;

	RegisterAllocation *m_allocation = nullptr;
	robin_hood::unordered_map<std::string, Bits> m_registerBits;
};

}
}
