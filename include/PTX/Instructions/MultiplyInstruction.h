#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MultiplyInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(MultiplyInstruction, ScalarType);
	DISABLE_TYPE(MultiplyInstruction, Int8Type);
	DISABLE_TYPE(MultiplyInstruction, UInt8Type);
public:
	MultiplyInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	void SetLower(bool lower)
	{
		m_lower = lower;
		if (lower)
		{
			m_upper = false;
		}
	}

	void SetUpper(bool upper)
	{
		m_upper = upper;
		if (upper)
		{
			m_lower = false;
		}
	}

	std::string OpCode() const
	{
		return "mul" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;

	bool m_upper = false;
	bool m_lower = false;
};

template<Bits B>
class MultiplyInstruction<FloatType<B>> : public PredicatedInstruction
{
public:
	MultiplyInstruction(Register<FloatType<B>> *destination, Operand<FloatType<B>> *sourceA, Operand<FloatType<B>> *sourceB, typename FloatType<B>::RoundingMode roundingMode = FloatType<B>::RoundingMode::None) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_roundingMode(roundingMode) {}

	void SetRoundingMode(typename FloatType<B>::RoundingMode roundingMode) { m_roundingMode = roundingMode; }
	void SetFlushSubNormal(bool flush) { m_flush = flush; }
	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		return "mul" + FloatType<B>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B>::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}
private:
	Register<FloatType<B>> *m_destination = nullptr;
	Operand<FloatType<B>> *m_sourceA = nullptr;
	Operand<FloatType<B>> *m_sourceB = nullptr;

	typename FloatType<B>::RoundingMode m_roundingMode = FloatType<B>::RoundingMode::None;
	bool m_flush = false;
	bool m_saturate = false;
};

template<>
class MultiplyInstruction<Float64Type> : public PredicatedInstruction
{
public:
	MultiplyInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_roundingMode(roundingMode) {}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	std::string OpCode() const
	{
		return "mul" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}
private:
	Register<Float64Type> *m_destination = nullptr;
	Operand<Float64Type> *m_sourceA = nullptr;
	Operand<Float64Type> *m_sourceB = nullptr;

	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
