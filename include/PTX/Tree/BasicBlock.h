#pragma once

#include <string>
#include <vector>

#include "PTX/Tree/Statements/StatementList.h"

#include "PTX/Tree/Operands/Label.h"
#include "PTX/Tree/Statements/Statement.h"

namespace PTX {

class BasicBlock : public StatementList
{
public:
	BasicBlock(Label *label) : m_label(label) {}

	// Labels

	const Label *GetLabel() const { return m_label; }
	Label *GetLabel() { return m_label; }

	void SetLabel(Label *label) { m_label = label; }

 	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::BasicBlock";
		j["statements"] = StatementList::ToJSON();
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& statement : m_statements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& statement : m_statements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	Label *m_label = nullptr;
};

}
