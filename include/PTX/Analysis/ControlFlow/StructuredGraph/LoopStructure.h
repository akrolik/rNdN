#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructureNode.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

namespace PTX {
namespace Analysis {

class LoopStructure : public StructureNode
{
public:
	LoopStructure(StructureNode *body, StructureNode *next) : StructureNode(next), m_body(body) {}

	// Loop body

	const StructureNode *GetBody() const { return m_body; }
	StructureNode *GetBody() { return m_body; }

	void SetBody(StructureNode *body) { m_body = body; }

	// Visitor

	void Accept(ConstStructuredGraphVisitor& visitor) const { visitor.Visit(this); }
	void Accept(StructuredGraphVisitor& visitor) { visitor.Visit(this); }
	
private:
	StructureNode *m_body = nullptr;
};

}
}
