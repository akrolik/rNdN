#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class DivideInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE(DivideInstruction,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float32Type, Float64Type
	);

	using InstructionBase_2<T>::InstructionBase;

	std::string OpCode() const override
	{
		return "div" + T::Name();
	}
};

template<>
class DivideInstruction<Float32Type> : public InstructionBase_2<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	DivideInstruction(const Register<Float32Type> *destination, const TypedOperand<Float32Type> *sourceA, TypedOperand<Float32Type> *sourceB, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_2<Float32Type>(destination, sourceA, sourceB), RoundingModifier<Float32Type>(roundingMode) {}

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

	std::string OpCode() const override
	{
		std::string code = "div";
		if (RoundingModifier<Float32Type>::IsActive())
		{
			code += RoundingModifier<Float32Type>::OpCodeModifier();
		}
		else if (m_full)
		{
			code += "full";
		}
		else
		{
			code += ".approx";
		}
		return code + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
	}

private:
	bool m_full = false;
};

template<>
class DivideInstruction<Float64Type> : public InstructionBase_2<Float64Type>, public RoundingModifier<Float64Type, true>
{
public:
	DivideInstruction(const Register<Float64Type> *destination, const TypedOperand<Float64Type> *sourceA, const TypedOperand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode) : InstructionBase_2<Float64Type>(destination, sourceA, sourceB), RoundingModifier<Float64Type, true>(roundingMode) {}

	std::string OpCode() const override
	{
		return "div" + RoundingModifier<Float64Type, true>::OpCodeModifier() + Float64Type::Name();
	}
};

}
