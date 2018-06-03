#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

class CallExpression : public Expression
{
public:
	CallExpression(std::string name, std::vector<Expression *> parameters) : m_name(name), m_parameters(parameters) {}

	std::string ToString() const
	{
		std::string code = m_name + "(";
		bool first = true;
		for (auto it = m_parameters.cbegin(); it != m_parameters.cend(); ++it)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += (*it)->ToString();
		}
		return code + ")";
	}

private:
	std::string m_name;
	std::vector<Expression *> m_parameters;
};

}
