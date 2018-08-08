#pragma once

#include <string>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Declaration.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class AssignStatement : public Statement
{
public:
	AssignStatement(const std::string& name, Type *type, Expression *expression) : m_declaration(new Declaration(name, type)), m_expression(expression) {}

	Declaration *GetDeclaration() const { return m_declaration; }
	Expression *GetExpression() const { return m_expression; }

	std::string ToString() const override
	{
		return m_declaration->ToString() + " = " + m_expression->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	Declaration *m_declaration = nullptr;
	Expression *m_expression = nullptr;
};

}
