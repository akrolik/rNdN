#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class VectorLiteral : public Literal
{
public:
	BasicType::BasicKind GetBasicKind() const { return m_basicKind; }

	// Properties

	virtual unsigned int GetCount() const = 0;

	// Formatting

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

	TypedVectorLiteral(const T& value, BasicType::BasicKind basicKind) : VectorLiteral(basicKind), m_values({value}) {}
	TypedVectorLiteral(const std::vector<T>& values, BasicType::BasicKind basicKind) : VectorLiteral(basicKind), m_values(values) {}

	// Values

	unsigned int GetCount() const override { return m_values.size(); }

	const std::vector<T>& GetValues() const { return m_values; }
	std::vector<T>& GetValues() { return m_values; }

	const T& GetValue(unsigned int index) const { return m_values.at(index); }
	T& GetValue(unsigned int index) { return m_values.at(index); }

	void SetValues(const std::vector<T>& values) { m_values = values; }

protected:
	std::vector<T> m_values;
};

template<class T>
class TypedVectorLiteral<T*> : public VectorLiteral
{
public:
	using Type = T*;

	TypedVectorLiteral(T *value, BasicType::BasicKind basicKind) : VectorLiteral(basicKind), m_values({value}) {}
	TypedVectorLiteral(const std::vector<T*>& values, BasicType::BasicKind basicKind) : VectorLiteral(basicKind), m_values(values) {}

	// Values

	std::vector<const T*> GetValues() const
	{
		return { std::begin(m_values), std::end(m_values) };
	}
	std::vector<T*>& GetValues() { return m_values; }

	const T *GetValue(unsigned int index) const { return m_values.at(index); }
	T *GetValue(unsigned int index) { return m_values.at(index); }

	void SetValues(const std::vector<T*>& values) { m_values = values; }

	unsigned int GetCount() const override { return m_values.size(); }

protected:
	std::vector<T*> m_values;
};

}
