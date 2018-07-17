#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class CallExpression : public Expression
{
public:
	CallExpression(std::string name, std::vector<Expression *> arguments) : m_name(name), m_arguments(arguments) {}

	const Type *GetType() const { return nullptr; }

	std::string GetName() const { return m_name; }
	const std::vector<Expression *>& GetArguments() const { return m_arguments; }
	Expression *GetArgument(unsigned int index) const { return m_arguments.at(index); }

	std::string ToString() const override
	{
		std::string code = m_name + "(";
		bool first = true;
		for (const auto& argument : m_arguments)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += argument->ToString();
		}
		return code + ")";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
	std::vector<Expression *> m_arguments;
};

}
