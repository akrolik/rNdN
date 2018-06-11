#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class CallExpression : public Expression
{
public:
	CallExpression(std::string name, std::vector<Expression *> parameters) : m_name(name), m_parameters(parameters) {}

	std::string GetName() const { return m_name; }
	const std::vector<Expression *>& GetParameters() const { return m_parameters; }

	std::string ToString() const override
	{
		std::string code = m_name + "(";
		bool first = true;
		for (auto parameter : m_parameters)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += parameter->ToString();
		}
		return code + ")";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_name;
	std::vector<Expression *> m_parameters;
};

}
