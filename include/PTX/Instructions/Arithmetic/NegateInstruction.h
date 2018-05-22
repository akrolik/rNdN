#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Arithmetic/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

template<class T>
class NegateInstruction : public InstructionBase<T, 1>
{
	REQUIRE_TYPE(NegateInstruction, ScalarType);
	DISABLE_TYPE(NegateInstruction, Int8Type);
	DISABLE_TYPES(NegateInstruction, UIntType);
	DISABLE_TYPE(NegateInstruction, Float16Type); //TODO: Missing from PTX specification
	DISABLE_TYPE(NegateInstruction, Float16x2Type); //TODO: Missing from PTX specification
public:
	using InstructionBase<T, 1>::InstructionBase;

	std::string OpCode() const
	{
		return "neg" + T::Name();
	}
};

template<>
class NegateInstruction<Float32Type> : public InstructionBase<Float32Type, 1>, public FlushSubnormalModifier
{
public:
	using InstructionBase<Float32Type, 1>::InstructionBase;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "neg.ftz" + Float32Type::Name();
		}
		return "neg" + Float32Type::Name();
	}
};

}
