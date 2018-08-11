#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

template<class T>
class Literal : public Expression
{
public:
	Literal(const T& value, Type *type) : m_values({value}) { SetType(type); }
	Literal(const std::vector<T>& values, Type *type) : m_values(values) { SetType(type); }

	const std::vector<T>& GetValues() const { return m_values; }
	const T& GetValue(unsigned int index) const { return m_values.at(index); }

	unsigned int GetCount() const { return m_values.size(); }

	std::string ToString() const override
	{
		std::string code;
		if (m_values.size() > 1)
		{
			code += "(";
		}
		bool first = true;
		for (const auto& value : m_values)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += std::to_string(value);
		}
		if (m_values.size() > 1)
		{
			code += ")";
		}
		return code + ":" + m_type->ToString();
	}

protected:
	std::vector<T> m_values;
};

template<>
inline std::string Literal<std::string>::ToString() const
{
	std::string code;
	if (m_values.size() > 1)
	{
		code += "(";
	}
	bool first = true;
	for (const auto& value : m_values)
	{
		if (!first)
		{
			code += ", ";
		}
		code += value;
	}
	if (m_values.size() > 1)
	{
		code += ")";
	}
	return code + ":" + m_type->ToString();
}

}
