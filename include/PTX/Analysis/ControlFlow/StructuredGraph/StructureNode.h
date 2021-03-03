#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

namespace PTX {
namespace Analysis {

class StructureNode
{
public:
	// Next structure

	const StructureNode *GetNext() const { return m_next; }
	StructureNode *GetNext() { return m_next; }

	void SetNext(StructureNode *next) { m_next = next; }

	// Visitor

	virtual void Accept(ConstStructuredGraphVisitor& visitor) const = 0;
	virtual void Accept(StructuredGraphVisitor& visitor) = 0;
	
protected:
	StructureNode(StructureNode *next) : m_next(next) { }

	StructureNode *m_next = nullptr;
};

}
}
