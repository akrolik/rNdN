#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class SubtractInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(SubtractInstruction, ScalarType);
	DISABLE_TYPE(SubtractInstruction, Int8Type);
	DISABLE_TYPE(SubtractInstruction, UInt8Type);
public:
	SubtractInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const
	{
		return "sub" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
};

template<>
class SubtractInstruction<Int32Type> : public PredicatedInstruction
{
	SubtractInstruction(Register<Int32Type> *destination, Operand<Int32Type> *sourceA, Operand<Int32Type> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		if (m_saturate)
		{
			return "sub.sat" + Int32Type::Name();

		}
		return "sub" + Int32Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<Int32Type> *m_destination = nullptr;
	Operand<Int32Type> *m_sourceA = nullptr;
	Operand<Int32Type> *m_sourceB = nullptr;

	bool m_saturate = false;
};

template<Bits B>
class SubtractInstruction<FloatType<B>> : public PredicatedInstruction
{
public:
	SubtractInstruction(Register<FloatType<B>> *destination, Operand<FloatType<B>> *sourceA, Operand<FloatType<B>> *sourceB, typename FloatType<B>::RoundingMode roundingMode = FloatType<B>::RoundingMode::None) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_roundingMode(roundingMode) {}

	void SetRoundingMode(typename FloatType<B>::RoundingMode roundingMode) { m_roundingMode = roundingMode; }
	void SetFlushSubNormal(bool flush) { m_flush = flush; }
	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		return "sub" + FloatType<B>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B>::Name();
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
class SubtractInstruction<Float64Type> : public PredicatedInstruction
{
public:
	SubtractInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_roundingMode(roundingMode) {}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	std::string OpCode() const
	{
		return "sub" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
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
