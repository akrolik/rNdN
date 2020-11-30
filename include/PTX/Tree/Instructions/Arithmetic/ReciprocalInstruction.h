#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"

namespace PTX {

template<class T, bool Assert = true>
class ReciprocalInstruction : public InstructionBase_1<T>
{
public:
	REQUIRE_TYPE_PARAM(ReciprocalInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);

	static std::string Mnemonic() { return "rcp"; }
};

template<>
class ReciprocalInstruction<Float32Type> : public InstructionBase_1<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	ReciprocalInstruction(const Register<Float32Type> *destination, const TypedOperand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_1<Float32Type>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	static std::string Mnemonic() { return "rcp"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if (m_roundingMode == Float32Type::RoundingMode::None)
		{
			code += ".approx";
		}
		else
		{
			code += RoundingModifier<Float32Type>::OpCodeModifier();
		}
		return code + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
	}
};

template<>
class ReciprocalInstruction<Float64Type> : public InstructionBase_1<Float64Type>, public RoundingModifier<Float64Type>
{
public:
	ReciprocalInstruction(const Register<Float64Type> *destination, const TypedOperand<Float64Type> *source, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : InstructionBase_1<Float64Type>(destination, source), RoundingModifier<Float64Type>(roundingMode) {}

	static std::string Mnemonic() { return "rcp"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			code += ".approx.ftz";
		}
		else
		{
			code += RoundingModifier<Float64Type>::OpCodeModifier();
		}
		return code + Float64Type::Name();
	}
};

}
