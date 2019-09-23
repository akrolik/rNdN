#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Tree/FunctionDeclaration.h"
#include "HorseIR/Tree/Expressions/Operand.h"
#include "HorseIR/Tree/Expressions/Literals/FunctionLiteral.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class CallExpression : public Expression
{
public:
	CallExpression(FunctionLiteral *literal, const std::vector<Operand *>& arguments) : m_literal(literal), m_arguments(arguments) {}

	CallExpression *Clone() const override
	{
		std::vector<Operand *> arguments;
		for (const auto& argument : m_arguments)
		{
			arguments.push_back(argument->Clone());
		}
		return new CallExpression(m_literal->Clone(), arguments);
	}

	FunctionLiteral *GetFunctionLiteral() const { return m_literal; }
	void SetFunctionLiteral(FunctionLiteral *literal) { m_literal = literal; }

	const std::vector<Operand *>& GetArguments() const { return m_arguments; }
	Operand *GetArgument(unsigned int index) const { return m_arguments.at(index); }
	void SetArguments(const std::vector<Operand *>& arguments) { m_arguments = arguments; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_literal->Accept(visitor);
			for (auto& argument : m_arguments)
			{
				argument->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_literal->Accept(visitor);
			for (const auto& argument : m_arguments)
			{
				argument->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	FunctionLiteral *m_literal = nullptr;
	std::vector<Operand *> m_arguments;
};

}
