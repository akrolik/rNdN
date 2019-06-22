#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class VectorLiteral : public Literal
{
public:
	BasicType::BasicKind GetBasicKind() const { return m_basicKind; }

	virtual unsigned int GetCount() const = 0;

	bool operator==(const VectorLiteral& other) const;
	bool operator!=(const VectorLiteral& other) const;

protected:
	VectorLiteral(BasicType::BasicKind basicKind) : Literal(Literal::Kind::Vector), m_basicKind(basicKind) {}

	BasicType::BasicKind m_basicKind;
};

template<class T>
class TypedVectorLiteral : public VectorLiteral
{
public:
	using Type = T;

	TypedVectorLiteral(const T& value, BasicType::BasicKind basicKind) : VectorLiteral(basicKind),  m_values({value}) {}
	TypedVectorLiteral(const std::vector<T>& values, BasicType::BasicKind basicKind) : VectorLiteral(basicKind), m_values(values) {}

	const std::vector<T>& GetValues() const { return m_values; }
	const T& GetValue(unsigned int index) const { return m_values.at(index); }
	void SetValues(const std::vector<T>& values) { m_values = values; }

	unsigned int GetCount() const override { return m_values.size(); }

protected:
	std::vector<T> m_values;
};

}
