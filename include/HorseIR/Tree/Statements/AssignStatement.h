#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Statements/Statement.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class AssignStatement : public Statement
{
public:
	AssignStatement(std::string identifier, Type *type, Expression *expression) : m_identifier(identifier), m_type(type), m_expression(expression) {}

	std::string ToString() const
	{
		return m_identifier + ":" + m_type->ToString() + " = " + m_expression->ToString();
	}

private:
	std::string m_identifier;
	Type *m_type = nullptr;
	Expression *m_expression = nullptr;
};

}
