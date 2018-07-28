#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class CastExpression : public Expression
{
public:
	CastExpression(Expression *expression, Type *type) : m_expression(expression), m_type(type) {}

	const Expression *GetExpression() const { return m_expression; }
	const Type *GetType() const { return m_type; }

	std::string ToString() const override
	{
		return "check_cast(" + m_expression->ToString() + ", " + m_type->ToString() + ")";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	Expression *m_expression = nullptr;
	Type *m_type = nullptr;
};

}
