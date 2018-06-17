#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

template<class T>
class Literal : public Expression
{
public:
	Literal(T value, Type *type) : m_values({value}), m_type(type) {}
	Literal(std::vector<T> values, Type *type) : m_values(values), m_type(type) {}

	const std::vector<T>& GetValues() const { return m_values; }
	const T& GetValue(unsigned int index) const { return m_values.at(index); }

	unsigned int GetCount() const { return m_values.size(); }

	std::string ToString() const override
	{
		std::string code = "(";
		bool first = true;
		for (auto value : m_values)
		{
			if (!first)
			{
				code += ", ";
			}
			code += std::to_string(value);
		}
		return code + "):" + m_type->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

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
