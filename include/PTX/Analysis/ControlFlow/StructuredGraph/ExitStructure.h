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
	ExitStructure(const BasicBlock *block, const Register<PredicateType> *predicate, StructureNode *next)
		: StructureNode(next), m_block(block), m_predicate(predicate) {}

	// Basic block

	const BasicBlock *GetBlock() const { return m_block; }
	void SetBlock(const BasicBlock *block) { m_block = block; }
	
	// Branch predicate

	const Register<PredicateType> *GetPredicate() const { return m_predicate; }
	void SetPredicate(const Register<PredicateType> *predicate) { m_predicate = predicate; }

	// Visitor

	void Accept(ConstStructuredGraphVisitor& visitor) const { visitor.Visit(this); }
	void Accept(StructuredGraphVisitor& visitor) { visitor.Visit(this); }
	
private:
	const BasicBlock *m_block = nullptr;
	const Register<PredicateType> *m_predicate = nullptr;
};

}
}
