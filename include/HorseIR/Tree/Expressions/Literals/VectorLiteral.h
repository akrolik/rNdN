#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

namespace HorseIR {

class VectorLiteral : public Literal
{
public:
	virtual unsigned int GetCount() const = 0;

protected:
	VectorLiteral() {}
};

template<class T>
class TypedVectorLiteral : public VectorLiteral
{
public:
	TypedVectorLiteral(const T& value) : m_values({value}) {}
	TypedVectorLiteral(const std::vector<T>& values) : m_values(values) {}

	const std::vector<T>& GetValues() const { return m_values; }
	const T& GetValue(unsigned int index) const { return m_values.at(index); }
	void SetValues(const std::vector<T>& values) { m_values = values; }

	unsigned int GetCount() const override { return m_values.size(); }

	bool operator==(const TypedVectorLiteral<T>& other) const
	{
		return (m_values == other.m_values);
	}

	bool operator!=(const TypedVectorLiteral<T>& other) const
	{
		return !(*this == other);
	}

protected:
	std::vector<T> m_values;
};

}
