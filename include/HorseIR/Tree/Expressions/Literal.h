#pragma once

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

template<class T>
class Literal : public Expression
{
public:
	Literal(T value, Type *type) : m_values({value}), m_type(type) {}
	Literal(std::vector<T> values, Type *type) : m_values(values), m_type(type) {}

	std::string ToString() const
	{
		std::string code = "(";
		bool first = true;
		for (auto it = m_values.cbegin(); it != m_values.cend(); ++it)
		{
			if (!first)
			{
				code += ", ";
			}
			code += std::to_string(*it);
		}
		return code + "):" + m_type->ToString();
	}

private:
	std::vector<T> m_values;
	Type *m_type = nullptr;
};

template<>
inline std::string Literal<std::string>::ToString() const
{
	std::string code = "(";
	bool first = true;
	for (auto it = m_values.cbegin(); it != m_values.cend(); ++it)
	{
		if (!first)
		{
			code += ", ";
		}
		code += *it;
	}
	return code + "):" + m_type->ToString();
}

}
