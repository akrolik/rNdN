#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class DivideInstruction : public InstructionBase_2<T>
{
	REQUIRE_BASE_TYPE(DivideInstruction, ScalarType);
	DISABLE_EXACT_TYPE(DivideInstruction, Int8Type);
	DISABLE_EXACT_TYPE(DivideInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(DivideInstruction, Float16Type);
	DISABLE_EXACT_TYPE(DivideInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(DivideInstruction, BitType);
public:
	using InstructionBase_2<T>::InstructionBase;

	std::string OpCode() const
	{
		return "div" + T::Name();
	}
};

template<>
class DivideInstruction<Float32Type> : public InstructionBase_2<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	DivideInstruction(Register<Float32Type> *destination, Operand<Float32Type> *sourceA, Operand<Float32Type> *sourceB, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_2<Float32Type>(destination, sourceA, sourceB), RoundingModifier<Float32Type>(roundingMode) {}

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
		std::string code = "div";
		if (this->m_roundingMode != Float32Type::RoundingMode::None)
		{
			code += Float32Type::RoundingModeString(this->m_roundingMode);
		}
		else if (m_full)
		{
			code += "full";
		}
		else
		{
			code += ".approx";
		}
		if (m_flush)
		{
			code += ".ftz";
		}
		return code + Float32Type::Name();
	}

private:
	bool m_full = false;
};

template<>
class DivideInstruction<Float64Type> : public InstructionBase_2<Float64Type>, public RoundingModifier<Float64Type, true>
{
public:
	DivideInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode) : InstructionBase_2<Float64Type>(destination, sourceA, sourceB), RoundingModifier<Float64Type, true>(roundingMode) {}

	std::string OpCode() const
	{
		return "div" + Float64Type::RoundingModeString(this->m_roundingMode) + Float64Type::Name();
	}
};

}
