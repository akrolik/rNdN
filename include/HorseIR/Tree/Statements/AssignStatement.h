#pragma once

#include <string>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class AssignStatement : public Statement
{
public:
	AssignStatement(std::string identifier, Type *type, Expression *expression) : m_identifier(identifier), m_type(type), m_expression(expression) {}

	std::string GetIdentifier() const { return m_identifier; }
	Type *GetType() const { return m_type; }
	Expression *GetExpression() const { return m_expression; }

	std::string ToString() const override
	{
		return m_identifier + ":" + m_type->ToString() + " = " + m_expression->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_identifier;
	Type *m_type = nullptr;
	Expression *m_expression = nullptr;
};

}
