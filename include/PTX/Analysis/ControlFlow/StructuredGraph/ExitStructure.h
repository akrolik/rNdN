#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructureNode.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class ExitStructure : public StructureNode
{
public:
	ExitStructure(const BasicBlock *block, const Register<PredicateType> *predicate, bool negate, StructureNode *next)
		: StructureNode(next), m_block(block), m_predicate(predicate), m_negate(negate) {}

	// Basic block

	const BasicBlock *GetBlock() const { return m_block; }
	void SetBlock(const BasicBlock *block) { m_block = block; }
	
	// Branch predicate

	std::pair<const Register<PredicateType> *, bool> GetPredicate() const { return { m_predicate, m_negate }; }
	void SetPredicate(const Register<PredicateType> *predicate, bool negate)
	{
		m_predicate = predicate;
		m_negate = negate;
	}

	// Visitor

	void Accept(ConstStructuredGraphVisitor& visitor) const { visitor.Visit(this); }
	void Accept(StructuredGraphVisitor& visitor) { visitor.Visit(this); }
	
private:
	const BasicBlock *m_block = nullptr;
	const Register<PredicateType> *m_predicate = nullptr;
	bool m_negate = false;
};

}
}
