#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

template<class T>
class Literal : public Expression
{
public:
	Literal(const T& value, BasicType *literalType) : m_values({value}), m_literalType(literalType) {}
	Literal(const std::vector<T>& values, BasicType *literalType) : m_values(values), m_literalType(literalType) {}

	const BasicType *GetLiteralType() const { return m_literalType; }

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
		return code + ":" + m_literalType->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	std::vector<T> m_values;
	BasicType *m_literalType = nullptr;
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
	return code + ":" + m_literalType->ToString();
}

}
