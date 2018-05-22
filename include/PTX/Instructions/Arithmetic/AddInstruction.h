#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Arithmetic/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Arithmetic/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Arithmetic/Modifiers/SaturateModifier.h"

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
class AddInstruction<Int32Type> : public InstructionBase<Int32Type, 2>, public SaturateModifier
{
public:
	using InstructionBase<Int32Type, 2>::InstructionBase;

	std::string OpCode() const
	{
		if (m_saturate)
		{
			return "add.sat" + Int32Type::Name();
		}
		return "add" + Int32Type::Name();
	}
};

template<Bits B, unsigned int N>
class AddInstruction<FloatType<B, N>> : public InstructionBase<FloatType<B, N>, 2>, public RoundingModifier<FloatType<B, N>>, public FlushSubnormalModifier, public SaturateModifier
{
public:
	AddInstruction(Register<FloatType<B, N>> *destination, Operand<FloatType<B, N>> *sourceA, Operand<FloatType<B, N>> *sourceB, typename FloatType<B, N>::RoundingMode roundingMode = FloatType<B, N>::RoundingMode::None) : InstructionBase<FloatType<B, N>, 2>(destination, sourceA, sourceB), RoundingModifier<FloatType<B, N>>(roundingMode) {}

	std::string OpCode() const
	{
		return "add" + FloatType<B, N>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B>::Name();
	}

private:
	using RoundingModifier<FloatType<B, N>>::m_roundingMode;
};

template<>
class AddInstruction<Float64Type> : public InstructionBase<Float64Type, 2>, public RoundingModifier<Float64Type>
{
public:
	AddInstruction(Register<Float64Type> *destination, Operand<Float64Type> *sourceA, Operand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : InstructionBase<Float64Type, 2>(destination, sourceA, sourceB), RoundingModifier<Float64Type>(roundingMode) {}

	std::string OpCode() const
	{
		return "add" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type>::m_roundingMode;
};

}
