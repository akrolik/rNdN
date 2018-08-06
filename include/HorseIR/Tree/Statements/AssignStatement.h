#pragma once

#include <string>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class AssignStatement : public Statement
{
public:
	AssignStatement(std::string targetName, Type *type, Expression *expression) : m_targetName(targetName), m_type(type), m_expression(expression) {}

	std::string GetTargetName() const { return m_targetName; }
	Type *GetType() const { return m_type; }
	Expression *GetExpression() const { return m_expression; }

	std::string ToString() const override
	{
		return m_targetName + ":" + m_type->ToString() + " = " + m_expression->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	std::string m_targetName;
	Type *m_type = nullptr;
	Expression *m_expression = nullptr;
};

}
