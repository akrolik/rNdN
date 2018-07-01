#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"

namespace PTX {

template<class T, bool Assert = true>
class RootInstruction : public InstructionBase_1<T>
{
public:
	REQUIRE_TYPE_PARAM(RootInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);
};

template<>
class RootInstruction<Float32Type> : public InstructionBase_1<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	RootInstruction(const Register<Float32Type> *destination, const TypedOperand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_1<Float32Type>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	std::string OpCode() const override
	{
		if (RoundingModifier<Float32Type>::IsActive())
		{
			return "sqrt" + RoundingModifier<Float32Type>::OpCodeModifier() + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
		}
		else
		{
			return "sqrt.approx" + FlushSubnormalModifier<Float32Type>::OpCodeModifier()+ Float32Type::Name();
		}
	}
};

template<>
class RootInstruction<Float64Type> : public InstructionBase_1<Float64Type>, RoundingModifier<Float64Type, true>
{
public:
	RootInstruction(const Register<Float64Type> *destination, const TypedOperand<Float64Type> *source, Float64Type::RoundingMode roundingMode) : InstructionBase_1<Float64Type>(destination, source), RoundingModifier<Float64Type, true>(roundingMode) {}

	std::string OpCode() const override
	{
		return "sqrt" + RoundingModifier<Float64Type, true>::OpCodeModifier() + Float64Type::Name();
	}
};

}
