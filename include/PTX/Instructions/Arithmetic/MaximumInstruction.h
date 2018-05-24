#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class MaximumInstruction : public InstructionBase<T, 2>
{
	REQUIRE_BASE_TYPE(MaximumInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MaximumInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MaximumInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MaximumInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MaximumInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MaximumInstruction, BitType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "max" + T::Name();
	}
};

template<>
class MaximumInstruction<Float32Type> : public InstructionBase<Float32Type, 2>, public FlushSubnormalModifier
{
public:
	using InstructionBase<Float32Type, 2>::InstructionBase;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "max.ftz" + Float32Type::Name();
		}
		return "max" + Float32Type::Name();
	}
};

}
