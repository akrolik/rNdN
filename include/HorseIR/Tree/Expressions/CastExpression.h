#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

class CastExpression : public Expression
{
public:
	CastExpression(Expression *expression, Type *type) : m_expression(expression), m_type(type) {}

	std::string ToString() const
	{
		return "check_cast(" + m_expression->ToString() + ", " + m_type->ToString() + ")";
	}

private:
	Expression *m_expression = nullptr;
	Type *m_type = nullptr;
};

}
