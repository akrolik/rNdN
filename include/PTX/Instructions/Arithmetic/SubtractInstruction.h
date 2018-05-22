#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Arithmetic/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Arithmetic/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Arithmetic/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class SubtractInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(SubtractInstruction, ScalarType);
	DISABLE_TYPE(SubtractInstruction, Int8Type);
	DISABLE_TYPE(SubtractInstruction, UInt8Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "sub" + T::Name();
	}
};

template<>
class SubtractInstruction<Int32Type> : public InstructionBase<Int32Type, 2>, public SaturateModifier
{
public:
	using InstructionBase<Int32Type, 2>::InstructionBase;

	std::string OpCode() const
	{
		if (m_saturate)
		{
			return "sub.sat" + Int32Type::Name();

		}
		return "sub" + Int32Type::Name();
	}
};

template<Bits B, unsigned int N>
class SubtractInstruction<FloatType<B, N>> : public InstructionBase<FloatType<B, N>, 2>, public RoundingModifier<FloatType<B, N>>, public FlushSubnormalModifier, public SaturateModifier
{
public:
	SubtractInstruction(Register<FloatType<B, N>> *destination, Operand<FloatType<B, N>> *sourceA, Operand<FloatType<B, N>> *sourceB, typename FloatType<B, N>::RoundingMode roundingMode = FloatType<B, N>::RoundingMode::None) : InstructionBase<FloatType<B, N>, 2>(destination, sourceA, sourceB), RoundingModifier<FloatType<B, N>>(roundingMode) {}

	std::string OpCode() const
	{
		return "sub" + FloatType<B, N>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B, N>::Name();
	}

private:
	using RoundingModifier<FloatType<B, N>>::m_roundingMode;
};

template<>
class SubtractInstruction<Float64Type> : public InstructionBase<Float64Type, 2>, RoundingModifier<Float64Type>
{
public:
	SubtractInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : InstructionBase<Float64Type, 2>(destination, sourceA, sourceB), RoundingModifier<Float64Type>(roundingMode) {}

	std::string OpCode() const
	{
		return "sub" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type>::m_roundingMode;
};

}
