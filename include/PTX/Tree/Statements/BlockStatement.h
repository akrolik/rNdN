#pragma once

#include "PTX/Tree/Statements/Statement.h"
#include "PTX/Tree/Statements/StatementList.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

class BlockStatement : public Statement, public StatementList
{
public:
	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::BlockStatement";
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
};

}
