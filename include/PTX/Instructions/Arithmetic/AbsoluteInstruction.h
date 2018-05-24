#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class AbsoluteInstruction : public InstructionBase<T, 1>
{
	REQUIRE_BASE_TYPE(AbsoluteInstruction, ScalarType);
	DISABLE_EXACT_TYPE(AbsoluteInstruction, Int8Type);
	DISABLE_EXACT_TYPE(AbsoluteInstruction, Float16Type);
	DISABLE_EXACT_TYPE(AbsoluteInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(AbsoluteInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(AbsoluteInstruction, UIntType);
public:
	using InstructionBase<T, 1>::InstructionBase;

	std::string OpCode() const
	{
		return "abs" + T::Name();
	}
};

template<>
class AbsoluteInstruction<Float32Type> : public InstructionBase<Float32Type, 1>, public FlushSubnormalModifier
{
public:
	using InstructionBase<Float32Type, 1>::InstructionBase;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "abs.ftz" + Float32Type::Name();
		}
		return "abs" + Float32Type::Name();
	}
};

}
