#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T>
class MADInstruction : public InstructionBase<T, 3>, public HalfModifier
{
	REQUIRE_BASE_TYPE(MADInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MADInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MADInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MADInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MADInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MADInstruction, BitType);
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

template<>
class MADInstruction<Float32Type> : public InstructionBase<Float32Type, 3>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier, public SaturateModifier
{
public:
	using InstructionBase<Float32Type, 3>::InstructionBase;

	std::string OpCode() const
	{
		return "mad" + Float32Type::RoundingModeString(m_roundingMode) + ((m_flush) ? ".ftz" : "") + ((m_saturate) ? ".sat" : "") + Float32Type::Name();
	}

private:
	using RoundingModifier<Float32Type>::m_roundingMode;
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
