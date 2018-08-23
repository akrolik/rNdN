#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Operand.h"

namespace HorseIR {

template<class T>
class Literal : public Operand
{
public:
	Literal(const T& value, Type *type) : Operand(Operand::Kind::Literal), m_values({value}) { SetType(type); }
	Literal(const std::vector<T>& values, Type *type) : Operand(Operand::Kind::Literal), m_values(values) { SetType(type); }

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

	bool operator==(const Literal& other)
	{
		return m_values == other.m_values;
	}

	bool operator!=(const Literal& other)
	{
		return !(*this == other);
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
