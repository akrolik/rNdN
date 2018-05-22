#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class DivideInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(DivideInstruction, ScalarType);
	DISABLE_TYPE(DivideInstruction, Int8Type);
	DISABLE_TYPE(DivideInstruction, UInt8Type);
	DISABLE_TYPE(DivideInstruction, Float16Type);
	DISABLE_TYPE(DivideInstruction, Float16x2Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "div" + T::Name();
	}
};

template<>
class DivideInstruction<Float32Type> : public InstructionBase<Float32Type, 2>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier
{
public:
	DivideInstruction(Register<Float32Type> *destination, Operand<Float32Type> *sourceA, Operand<Float32Type> *sourceB, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase<Float32Type, 2>(destination, sourceA, sourceB), RoundingModifier<Float32Type>(roundingMode) {}

	void SetRoundingMode(Float32Type::RoundingMode roundingMode)
	{
		if (roundingMode != Float32Type::RoundingMode::None)
		{
			m_full = false;
		}
		m_roundingMode = roundingMode;
	}

	bool GetFull() const { return m_full; }
	void SetFull(bool full)
	{
		if (full)
		{
			m_roundingMode == Float32Type::RoundingMode::None;
		}
		m_full = full;
	}

	std::string OpCode() const
	{
		if (m_roundingMode == Float32Type::RoundingMode::None)
		{
			return "div" + Float32Type::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + Float32Type::Name();
		}
		else if (m_full)
		{
			return std::string("div.full") + ((m_flush) ? ".ftz" : "") + Float32Type::Name();
		}
		return std::string("div.approx") + ((m_flush) ? ".ftz" : "") + Float32Type::Name();
	}

private:
	using RoundingModifier<Float32Type>::m_roundingMode;
	bool m_full = false;
};

template<>
class DivideInstruction<Float64Type> : public InstructionBase<Float64Type, 2>, public RoundingModifier<Float64Type, true>
{
public:
	DivideInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode) : InstructionBase<Float64Type, 2>(destination, sourceA, sourceB), RoundingModifier<Float64Type, true>(roundingMode) {}

	std::string OpCode() const
	{
		return "div" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type, true>::m_roundingMode;
};

}
