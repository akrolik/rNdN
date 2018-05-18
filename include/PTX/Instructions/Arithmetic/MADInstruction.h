#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class MADInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(MADInstruction, ScalarType);
	DISABLE_TYPE(MADInstruction, Int8Type);
	DISABLE_TYPE(MADInstruction, UInt8Type);
public:
	MADInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<T> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	void SetLower(bool lower)
	{
		m_lower = lower;
		if (lower) { m_upper = false; }
	}

	void SetUpper(bool upper)
	{
		m_upper = upper;
		if (upper) { m_lower = false; }
	}

	std::string OpCode() const
	{
		if (m_lower)
		{
			return "mad.lo" + T::Name();
		}
		else if (m_upper)
		{
			return "mad.hi" + T::Name();
		}
		return "mad" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	Operand<T> *m_sourceC = nullptr;

	bool m_lower = false;
	bool m_upper = false;
};

template<>
class MADInstruction<Int32Type> : public PredicatedInstruction
{
public:
	MADInstruction(Register<Int32Type> *destination, Operand<Int32Type> *sourceA, Operand<Int32Type> *sourceB, Operand<Int32Type> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	void SetLower(bool lower)
	{
		m_lower = lower;
		if (lower) { m_upper = false; }
	}

	void SetUpper(bool upper)
	{
		m_upper = upper;
		if (upper) { m_lower = false; }
	}

	std::string OpCode() const
	{
		if (m_lower)
		{
			return "mad.lo" + Int32Type::Name();
		}
		else if (m_upper)
		{
			// Only applies in .hi mode
			if (m_saturate)
			{
				return "mad.hi.sat" + Int32Type::Name();
			}
			return "mad.hi" + Int32Type::Name();
		}

		if (m_saturate)
		{
			return "mad.sat" + Int32Type::Name();
		}
		return "mad" + Int32Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<Int32Type> *m_destination = nullptr;
	Operand<Int32Type> *m_sourceA = nullptr;
	Operand<Int32Type> *m_sourceB = nullptr;
	Operand<Int32Type> *m_sourceC = nullptr;

	bool m_lower = false;
	bool m_upper = false;
	bool m_saturate = false;
};


template<Bits B>
class MADInstruction<FloatType<B>> : public PredicatedInstruction
{
	DISABLE_BITS(MADInstruction, FloatType, Bits16);
public:
	MADInstruction(Register<FloatType<B>> *destination, Operand<FloatType<B>> *sourceA, Operand<FloatType<B>> *sourceB, Operand<FloatType<B>> *sourceC, typename FloatType<B>::RoundingMode roundingMode) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_roundingMode(roundingMode) {}

	void SetRoundingMode(typename FloatType<B>::RoundingMode roundingMode) { m_roundingMode = roundingMode; }
	void SetFlushSubNormal(bool flush) { m_flush = flush; }
	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		return "mad" + FloatType<B>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B>::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}
private:
	Register<FloatType<B>> *m_destination = nullptr;
	Operand<FloatType<B>> *m_sourceA = nullptr;
	Operand<FloatType<B>> *m_sourceB = nullptr;
	Operand<FloatType<B>> *m_sourceC = nullptr;

	typename FloatType<B>::RoundingMode m_roundingMode = FloatType<B>::RoundingMode::None;
	bool m_flush = false;
	bool m_saturate = false;
};

template<>
class MADInstruction<Float64Type> : public PredicatedInstruction
{
public:
	MADInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Operand<Float64Type> *sourceC, typename Float64Type::RoundingMode roundingMode) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_roundingMode(roundingMode) {}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	std::string OpCode() const
	{
		return "mad" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}
private:
	Register<Float64Type> *m_destination = nullptr;
	Operand<Float64Type> *m_sourceA = nullptr;
	Operand<Float64Type> *m_sourceB = nullptr;
	Operand<Float64Type> *m_sourceC = nullptr;

	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
