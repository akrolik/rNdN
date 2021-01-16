#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"

namespace PTX {

DispatchInterface(RootInstruction)

template<class T, bool Assert = true>
class RootInstruction : DispatchInherit(RootInstruction), public InstructionBase_1<T>
{
public:
	REQUIRE_TYPE_PARAM(RootInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);

	// Formatting

	static std::string Mnemonic() { return "sqrt"; }

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

template<>
class RootInstruction<Float32Type> : DispatchInherit(RootInstruction), public InstructionBase_1<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	RootInstruction(Register<Float32Type> *destination, TypedOperand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_1<Float32Type>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	// Formatting

	static std::string Mnemonic() { return "sqrt"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if (RoundingModifier<Float32Type>::IsActive())
		{
			code += RoundingModifier<Float32Type>::GetOpCodeModifier();
		}
		else
		{
			code += ".approx";
		}

		return code + FlushSubnormalModifier<Float32Type>::GetOpCodeModifier() + Float32Type::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float32Type);
};

template<>
class RootInstruction<Float64Type> : DispatchInherit(RootInstruction), public InstructionBase_1<Float64Type>, RoundingModifier<Float64Type, true>
{
public:
	RootInstruction(Register<Float64Type> *destination, TypedOperand<Float64Type> *source, Float64Type::RoundingMode roundingMode) : InstructionBase_1<Float64Type>(destination, source), RoundingModifier<Float64Type, true>(roundingMode) {}

	// Formatting

	static std::string Mnemonic() { return "sqrt"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + RoundingModifier<Float64Type, true>::GetOpCodeModifier() + Float64Type::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float64Type);
};

}
