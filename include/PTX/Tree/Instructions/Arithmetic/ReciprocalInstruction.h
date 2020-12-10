#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(ReciprocalInstruction)

template<class T, bool Assert = true>
class ReciprocalInstruction : DispatchInherit(ReciprocalInstruction), public InstructionBase_1<T>
{
public:
	REQUIRE_TYPE_PARAM(ReciprocalInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);

	// Formatting

	static std::string Mnemonic() { return "rcp"; }

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

template<>
class ReciprocalInstruction<Float32Type> : DispatchInherit(ReciprocalInstruction), public InstructionBase_1<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	ReciprocalInstruction(Register<Float32Type> *destination, TypedOperand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_1<Float32Type>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	// Formatting

	static std::string Mnemonic() { return "rcp"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if (m_roundingMode == Float32Type::RoundingMode::None)
		{
			code += ".approx";
		}
		else
		{
			code += RoundingModifier<Float32Type>::GetOpCodeModifier();
		}
		return code + FlushSubnormalModifier<Float32Type>::GetOpCodeModifier() + Float32Type::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float32Type);
};

template<>
class ReciprocalInstruction<Float64Type> : DispatchInherit(ReciprocalInstruction), public InstructionBase_1<Float64Type>, public RoundingModifier<Float64Type>
{
public:
	ReciprocalInstruction(Register<Float64Type> *destination, TypedOperand<Float64Type> *source, Float64Type::RoundingMode roundingMode = Float64Type::RoundingMode::None) : InstructionBase_1<Float64Type>(destination, source), RoundingModifier<Float64Type>(roundingMode) {}

	// Formatting

	static std::string Mnemonic() { return "rcp"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if (m_roundingMode == Float64Type::RoundingMode::None)
		{
			code += ".approx.ftz";
		}
		else
		{
			code += RoundingModifier<Float64Type>::GetOpCodeModifier();
		}
		return code + Float64Type::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float64Type);
};

DispatchImplementation(ReciprocalInstruction)

}
