#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Arithmetic/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Arithmetic/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Arithmetic/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Arithmetic/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class MADInstruction : public InstructionBase<T, 3>, public HalfModifier
{
	REQUIRE_TYPE(MADInstruction, ScalarType);
	DISABLE_TYPE(MADInstruction, Int8Type);
	DISABLE_TYPE(MADInstruction, UInt8Type);
public:
	using InstructionBase<T, 3>::InstructionBase;

	std::string OpCode() const
	{
		if (m_lower)
		{
			return "mad.lo" + T::Name();
		}
		else if (m_upper)
		{
			return "mad.hi" + T::Name();
		}
		return "mad" + T::Name();
	}
};

template<>
class MADInstruction<Int32Type> : public InstructionBase<Int32Type, 3>, public HalfModifier, public SaturateModifier
{
public:
	using InstructionBase<Int32Type, 3>::InstructionBase;

	std::string OpCode() const
	{
		if (m_lower)
		{
			return "mad.lo" + Int32Type::Name();
		}
		else if (m_upper)
		{
			// Only applies in .hi mode
			if (m_saturate)
			{
				return "mad.hi.sat" + Int32Type::Name();
			}
			return "mad.hi" + Int32Type::Name();
		}

		if (m_saturate)
		{
			return "mad.sat" + Int32Type::Name();
		}
		return "mad" + Int32Type::Name();
	}
};

template<Bits B, unsigned int N>
class MADInstruction<FloatType<B, N>> : public InstructionBase<FloatType<B, N>, 3>, public RoundingModifier<FloatType<B, N>>, public FlushSubnormalModifier, public SaturateModifier
{
	DISABLE_TYPE_BITS(MADInstruction, FloatType, Bits16);
public:
	using InstructionBase<FloatType<B, N>, 3>::InstructionBase;

	std::string OpCode() const
	{
		return "mad" + FloatType<B, N>::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + FloatType<B>::Name();
	}

private:
	using RoundingModifier<FloatType<B, N>>::m_roundingMode;
};

template<>
class MADInstruction<Float64Type> : public InstructionBase<Float64Type, 3>, public RoundingModifier<Float64Type>
{
public:
	using InstructionBase<Float64Type, 3>::InstructionBase;

	std::string OpCode() const
	{
		return "mad" + Float64Type::RoundingModeString(m_roundingMode) + Float64Type::Name();
	}

private:
	using RoundingModifier<Float64Type>::m_roundingMode;
};

}
