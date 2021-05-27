#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructureNode.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class BranchStructure : public StructureNode
{
public:
	BranchStructure(BasicBlock *block, Register<PredicateType> *predicate, StructureNode *trueBranch, StructureNode *falseBranch, StructureNode *next)
		: StructureNode(next), m_block(block), m_predicate(predicate), m_trueBranch(trueBranch), m_falseBranch(falseBranch) {}

	// Basic block

	const BasicBlock *GetBlock() const { return m_block; }
	BasicBlock *GetBlock() { return m_block; }
	void SetBlock(BasicBlock *block) { m_block = block; }

	// Branch predicate

	const Register<PredicateType> *GetPredicate() const { return m_predicate; }
	Register<PredicateType> *GetPredicate() { return m_predicate; }
	void SetPredicate(Register<PredicateType> *predicate) { m_predicate = predicate; }

	// True structure

	const StructureNode *GetTrueBranch() const { return m_trueBranch; }
	StructureNode *GetTrueBranch() { return m_trueBranch; }

	void SetTrueBranch(StructureNode *trueBranch) { m_trueBranch = trueBranch; }

	// False structure

	const StructureNode *GetFalseBranch() const { return m_falseBranch; }
	StructureNode *GetFalseBranch() { return m_falseBranch; }

	void SetFalseBranch(StructureNode *falseBranch) { m_falseBranch = falseBranch; }

	// Visitor

	void Accept(ConstStructuredGraphVisitor& visitor) const { visitor.Visit(this); }
	void Accept(StructuredGraphVisitor& visitor) { visitor.Visit(this); }
	
private:
	BasicBlock *m_block = nullptr;
	Register<PredicateType> *m_predicate = nullptr;
	StructureNode *m_trueBranch = nullptr;
	StructureNode *m_falseBranch = nullptr;
};

}
}
