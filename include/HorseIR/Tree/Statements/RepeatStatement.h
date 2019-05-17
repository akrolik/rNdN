#pragma once

#include <vector>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class RepeatStatement : public Statement
{
public:
	RepeatStatement(Operand *condition, const std::vector<Statement *>& body) : m_condition(condition), m_body(body) {}

	Operand *GetCondition() const { return m_condition; }
	void SetCondition(Operand *condition) { m_condition = condition; }

	const std::vector<Statement *>& GetBody() const { return m_body; }
	void SetBody(const std::vector<Statement *>& body) { m_body = body; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			for (auto& statement : m_body)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			for (const auto& statement : m_body)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	Operand *m_condition = nullptr;
	std::vector<Statement *> m_body;
};

}
