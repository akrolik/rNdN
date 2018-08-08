#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class CastExpression : public Expression
{
public:
	CastExpression(Expression *expression, Type *castType) : m_expression(expression), m_castType(castType) {}

	Expression *GetExpression() const { return m_expression; }
	Type *GetCastType() const { return m_castType; }

	std::string ToString() const override
	{
		return "check_cast(" + m_expression->ToString() + ", " + m_castType->ToString() + ")";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	Expression *m_expression = nullptr;
	Type *m_castType = nullptr;
};

}
