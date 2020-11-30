#pragma once

#include "PTX/Statements/Statement.h"
#include "PTX/Statements/StatementList.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

class BlockStatement : public Statement, public StatementList
{
public:
	std::string ToString(unsigned int indentation) const override
	{
		return StatementList::ToString(indentation);
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::BlockStatement";
		j["statements"] = StatementList::ToJSON();
		return j;
	}

	// Visitors

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
