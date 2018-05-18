#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class AddInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(AddInstruction, ScalarType);
	DISABLE_TYPE(AddInstruction, Int8Type);
	DISABLE_TYPE(AddInstruction, UInt8Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "add" + T::Name();
	}
};

template<>
class AddInstruction<Int32Type> : public InstructionBase<Int32Type, 2>
{
public:
	using InstructionBase<Int32Type, 2>::InstructionBase;

	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		if (m_saturate)
		{
			return "add.sat" + Int32Type::Name();

		}
		return "add" + Int32Type::Name();
	}

private:
	bool m_saturate = false;
};

template<Bits B>
class AddInstruction<FloatType<B>> : public InstructionBase<FloatType<B>, 2>
{
public:
	AddInstruction(Register<FloatType<B>> *destination, Operand<FloatType<B>> *sourceA, Operand<FloatType<B>> *sourceB, typename FloatType<B>::RoundingMode roundingMode = FloatType<B>::RoundingMode::None) : InstructionBase<FloatType<B>, 2>(destination, sourceA, sourceB), m_roundingMode(roundingMode) {}

	void SetRoundingMode(typename FloatType<B>::RoundingMode roundingMode) { m_roundingMode = roundingMode; }
	void SetFlushSubNormal(bool flush) { m_flush = flush; }
	void SetSaturate(bool saturate) { m_saturate = saturate; }

	std::string OpCode() const
	{
		return "add" + FloatType<B>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B>::Name();
	}

private:
	typename FloatType<B>::RoundingMode m_roundingMode = FloatType<B>::RoundingMode::None;
	bool m_flush = false;
	bool m_saturate = false;
};

template<>
class AddInstruction<Float64Type> : public InstructionBase<Float64Type, 2>
{
public:
	AddInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : InstructionBase<Float64Type, 2>(destination, sourceA, sourceB), m_roundingMode(roundingMode) {}

	void SetRoundingMode(Float64Type::RoundingMode roundingMode) { m_roundingMode = roundingMode; }

	std::string OpCode() const
	{
		return "add" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	Float64Type::RoundingMode m_roundingMode = Float64Type::RoundingMode::None;
};

}
