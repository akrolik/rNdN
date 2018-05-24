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
class RootInstruction<Float32Type> : public InstructionBase<Float32Type, 1>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier
{
public:
	RootInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase<Float32Type, 1>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	std::string OpCode() const
	{
		if (m_roundingMode == Float32Type::RoundingMode::None)
		{
			if (m_flush)
			{
				return "sqrt.approx.ftz" + Float32Type::Name();
			}
			return "sqrt.approx" + Float32Type::Name();
		}
		else
		{
			if (m_flush)
			{
				return "sqrt" + Float32Type::RoundingModeString(m_roundingMode) + ".ftz" + Float32Type::Name();
			}
			return "sqrt" + Float32Type::RoundingModeString(m_roundingMode) + Float32Type::Name();
		}
	}

private:
	using RoundingModifier<Float32Type>::m_roundingMode;
};

template<>
class RootInstruction<Float64Type> : public InstructionBase<Float64Type, 1>, RoundingModifier<Float64Type, true>
{
public:
	RootInstruction(Register<Float64Type> *destination, Operand<Float64Type> *source, Float64Type::RoundingMode roundingMode) : InstructionBase<Float64Type, 1>(destination, source), RoundingModifier<Float64Type, true>(roundingMode) {}

	std::string OpCode() const
	{
		return "rcp" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type, true>::m_roundingMode;
};

}
