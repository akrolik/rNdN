#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class CastExpression : public Expression
{
public:
	CastExpression(Expression *expression, Type *castType) : m_expression(expression), m_castType(castType) {}

	CastExpression *Clone() const override
	{
		return new CastExpression(m_expression->Clone(), m_castType->Clone());
	}

	Expression *GetExpression() const { return m_expression; }
	void SetExpression(Expression *expression) { m_expression = expression; }

	Type *GetCastType() const { return m_castType; }
	void SetCastType(Type *castType) { m_castType = castType; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_expression->Accept(visitor);
			m_castType->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_expression->Accept(visitor);
			m_castType->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	Expression *m_expression = nullptr;
	Type *m_castType = nullptr;
};

}
