#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructureNode.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class SequenceStructure : public StructureNode
{
public:
	SequenceStructure(const BasicBlock *block, StructureNode *next) : StructureNode(next), m_block(block) {}

	// Basic block

	const BasicBlock *GetBlock() const { return m_block; }
	void SetBlock(const BasicBlock *block) { m_block = block; }

	// Visitor

	void Accept(ConstStructuredGraphVisitor& visitor) const { visitor.Visit(this); }
	void Accept(StructuredGraphVisitor& visitor) { visitor.Visit(this); }
	
private:
	const BasicBlock *m_block = nullptr;
};

}
}
