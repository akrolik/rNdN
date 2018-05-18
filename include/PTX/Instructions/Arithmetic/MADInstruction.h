#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class MADInstruction : public InstructionBase<T, 3>
{
	REQUIRE_TYPE(MADInstruction, ScalarType);
	DISABLE_TYPE(MADInstruction, Int8Type);
	DISABLE_TYPE(MADInstruction, UInt8Type);
public:
	using InstructionBase<T, 3>::InstructionBase;

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

private:
	bool m_lower = false;
	bool m_upper = false;
};

template<>
class MADInstruction<Int32Type> : public InstructionBase<Int32Type, 3>
{
public:
	using InstructionBase<Int32Type, 3>::InstructionBase;

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

private:
	bool m_lower = false;
	bool m_upper = false;
	bool m_saturate = false;
};


template<Bits B>
class MADInstruction<FloatType<B>> : public InstructionBase<FloatType<B>, 3>
{
	DISABLE_TYPE_BITS(MADInstruction, FloatType, Bits16);
public:
	using InstructionBase<FloatType<B>, 3>::InstructionBase;

	void SetRoundingMode(typename FloatType<B>::RoundingMode roundingMode) { m_roundingMode = roundingMode; }
	void SetFlushSubNormal(bool flush) { m_flush = flush; }
	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		return "mad" + FloatType<B>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B>::Name();
	}

private:
	typename FloatType<B>::RoundingMode m_roundingMode = FloatType<B>::RoundingMode::None;
	bool m_flush = false;
	bool m_saturate = false;
};

template<>
class MADInstruction<Float64Type> : public InstructionBase<Float64Type, 3>
{
public:
	using InstructionBase<Float64Type, 3>::InstructionBase;

	void SetRoundingMode(Float64Type::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	std::string OpCode() const
	{
		return "mad" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
