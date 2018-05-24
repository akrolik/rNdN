#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class FMAInstruction : public InstructionBase<T, 3>, public RoundingModifier<T, true>, public FlushSubnormalModifier, public SaturateModifier
{
	REQUIRE_EXACT_TYPE_TEMPLATE(FMAInstruction, FloatType);
public:
	FMAInstruction(Register<T> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<T> *sourceC, typename T::RoundingMode roundingMode) : InstructionBase<T, 3>(destination, sourceA, sourceB, sourceC), RoundingModifier<T, true>(roundingMode) {}

	std::string OpCode() const
	{
		return "fma" + T::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + T::Name();
	}

private:
	using RoundingModifier<T, true>::m_roundingMode;
};

template<>
class FMAInstruction<Float64Type> : public InstructionBase<Float64Type, 3>, public RoundingModifier<Float64Type, true>
{
public:
	FMAInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Operand<Float64Type> *sourceC, Float64Type::RoundingMode roundingMode) : InstructionBase<Float64Type, 3>(destination, sourceA, sourceB, sourceC), RoundingModifier<Float64Type, true>(roundingMode) {}

	std::string OpCode() const
	{
		return "fma" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type, true>::m_roundingMode;
};

}
