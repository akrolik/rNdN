#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"

namespace PTX {

template<class T>
class ReciprocalInstruction : public InstructionBase<T, 1>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(ReciprocalInstruction, FloatType);
	DISABLE_EXACT_TYPE(ReciprocalInstruction, Float16Type);
	DISABLE_EXACT_TYPE(ReciprocalInstruction, Float16x2Type);
};

template<>
class ReciprocalInstruction<Float32Type> : public InstructionBase<Float32Type, 1>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier
{
public:
	ReciprocalInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase<Float32Type, 1>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	std::string OpCode() const
	{
		if (m_roundingMode == Float32Type::RoundingMode::None)
		{
			if (m_flush)
			{
				return "rcp.approx.ftz" + Float32Type::Name();
			}
			return "rcp.approx" + Float32Type::Name();
		}
		else
		{
			if (m_flush)
			{
				return "rcp" + Float32Type::RoundingModeString(m_roundingMode) + ".ftz" + Float32Type::Name();
			}
			return "rcp" + Float32Type::RoundingModeString(m_roundingMode) + Float32Type::Name();
		}
	}

private:
	using RoundingModifier<Float32Type>::m_roundingMode;
};

template<>
class ReciprocalInstruction<Float64Type> : public InstructionBase<Float64Type, 1>, public RoundingModifier<Float64Type>
{
public:
	ReciprocalInstruction(Register<Float64Type> *destination, Operand<Float64Type> *source, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : InstructionBase<Float64Type, 1>(destination, source), RoundingModifier<Float64Type>(roundingMode) {}

	std::string OpCode() const
	{
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			return "rcp.approx.ftz" + Float64Type::Name();
		}
		return "rcp" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type>::m_roundingMode;
};

}
