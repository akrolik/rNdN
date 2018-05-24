#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class MinimumInstruction : public InstructionBase<T, 2>
{
	REQUIRE_BASE_TYPE(MinimumInstruction, ScalarType);
	DISABLE_EXACT_TYPE(MinimumInstruction, Int8Type);
	DISABLE_EXACT_TYPE(MinimumInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(MinimumInstruction, Float16Type);
	DISABLE_EXACT_TYPE(MinimumInstruction, Float16x2Type);
	DISABLE_EXACT_TYPE_TEMPLATE(MinimumInstruction, BitType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "min" + T::Name();
	}
};

template<>
class MinimumInstruction<Float32Type> : public InstructionBase<Float32Type, 2>, public FlushSubnormalModifier
{
public:
	using InstructionBase<Float32Type, 2>::InstructionBase;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "min.ftz" + Float32Type::Name();
		}
		return "min" + Float32Type::Name();
	}
};

}
