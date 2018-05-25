#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"

namespace PTX {

template<class T>
class RootInstruction : public InstructionBase<T, 1>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(RootInstruction, FloatType);
	DISABLE_EXACT_TYPE(RootInstruction, Float16Type);
	DISABLE_EXACT_TYPE(RootInstruction, Float16x2Type);
};

template<>
class RootInstruction<Float32Type> : public InstructionBase<Float32Type, 1>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	RootInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase<Float32Type, 1>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	std::string OpCode() const
	{
		if (this->m_roundingMode == Float32Type::RoundingMode::None)
		{
			if (this->m_flush)
			{
				return "sqrt.approx.ftz" + Float32Type::Name();
			}
			return "sqrt.approx" + Float32Type::Name();
		}
		else
		{
			if (this->m_flush)
			{
				return "sqrt" + Float32Type::RoundingModeString(this->m_roundingMode) + ".ftz" + Float32Type::Name();
			}
			return "sqrt" + Float32Type::RoundingModeString(this->m_roundingMode) + Float32Type::Name();
		}
	}
};

template<>
class RootInstruction<Float64Type> : public InstructionBase<Float64Type, 1>, RoundingModifier<Float64Type, true>
{
public:
	RootInstruction(Register<Float64Type> *destination, Operand<Float64Type> *source, Float64Type::RoundingMode roundingMode) : InstructionBase<Float64Type, 1>(destination, source), RoundingModifier<Float64Type, true>(roundingMode) {}

	std::string OpCode() const
	{
		return "sqrt" + Float64Type::RoundingModeString(this->m_roundingMode) + Float64Type::Name();
	}
};

}
