#pragma once

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructureNode.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/ExitStructure.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

class LoopStructure : public StructureNode
{
public:
	LoopStructure(StructureNode *body, const robin_hood::unordered_set<ExitStructure *>& exits, StructureNode *latch, StructureNode *next) : StructureNode(next), m_body(body), m_exits(exits), m_latch(latch) {}

	// Loop body

	const StructureNode *GetBody() const { return m_body; }
	StructureNode *GetBody() { return m_body; }

	void SetBody(StructureNode *body) { m_body = body; }

	// Exits

	const robin_hood::unordered_set<ExitStructure *>& GetExits() { return m_exits; }
	robin_hood::unordered_set<const ExitStructure *> GetExits() const
	{
		return { std::begin(m_exits), std::end(m_exits) };
	}

	void AddExit(ExitStructure *exit) { m_exits.insert(exit); }
	void SetExits(const robin_hood::unordered_set<ExitStructure *>& exits) { m_exits = exits; }

	// Latch

	const StructureNode *GetLatch() const { return m_latch; }
	StructureNode *GetLatch() { return m_latch; }

	void SetLatch(StructureNode *latch) { m_latch = latch; }

	// Visitor

	void Accept(ConstStructuredGraphVisitor& visitor) const { visitor.Visit(this); }
	void Accept(StructuredGraphVisitor& visitor) { visitor.Visit(this); }
	
private:
	StructureNode *m_body = nullptr;
	robin_hood::unordered_set<ExitStructure *> m_exits;
	StructureNode *m_latch = nullptr;
};

}
}
