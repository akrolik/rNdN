#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class MultiplyInstruction : public InstructionBase<T, 2>, public HalfModifier
{
	REQUIRE_BASE_TYPE(MultiplyInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MultiplyInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MultiplyInstruction, UInt8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MultiplyInstruction, BitType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		if (m_upper)
		{
			return "mul.hi" + T::Name();
		}
		else if (m_lower)
		{
			return "mul.lo" + T::Name();
		}
		return "mul" + T::Name();
	}
};

template<Bits B, unsigned int N>
class MultiplyInstruction<FloatType<B, N>> : public InstructionBase<FloatType<B, N>, 2>, public RoundingModifier<FloatType<B, N>>, public FlushSubnormalModifier, public SaturateModifier
{
public:
	MultiplyInstruction(Register<FloatType<B, N>> *destination, Operand<FloatType<B, N>> *sourceA, Operand<FloatType<B, N>> *sourceB, typename FloatType<B, N>::RoundingMode roundingMode = FloatType<B, N>::RoundingMode::None) : InstructionBase<FloatType<B, N>, 2>(destination, sourceA, sourceB), RoundingModifier<FloatType<B, N>>(roundingMode) {}

	std::string OpCode() const
	{
		return "mul" + FloatType<B, N>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B, N>::Name();
	}

private:
	using RoundingModifier<FloatType<B, N>>::m_roundingMode;
};

template<>
class MultiplyInstruction<Float64Type> : public InstructionBase<Float64Type, 2>, public RoundingModifier<Float64Type>
{
public:
	MultiplyInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : InstructionBase<Float64Type, 2>(destination, sourceA, sourceB), RoundingModifier<Float64Type>(roundingMode) {}

	std::string OpCode() const
	{
		return "mul" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type>::m_roundingMode;
};

}
