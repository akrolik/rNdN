#pragma once

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class ExpressionStatement : public Statement
{
public:
	ExpressionStatement(Expression *expression) : m_expression(expression) {}

	ExpressionStatement *Clone() const override
	{
		return new ExpressionStatement(m_expression->Clone());
	}

	Expression *GetExpression() const { return m_expression; }
	void SetExpression(Expression *expression) { m_expression = expression; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	Expression *m_expression = nullptr;
};

}
