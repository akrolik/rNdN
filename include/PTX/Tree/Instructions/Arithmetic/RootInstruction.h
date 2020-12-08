#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	static std::string Mnemonic() { return "sqrt"; }

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

template<>
class RootInstruction<Float32Type> : DispatchInherit(RootInstruction), public InstructionBase_1<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	RootInstruction(const Register<Float32Type> *destination, const TypedOperand<Float32Type> *source, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_1<Float32Type>(destination, source), RoundingModifier<Float32Type>(roundingMode) {}

	static std::string Mnemonic() { return "sqrt"; }
	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if (RoundingModifier<Float32Type>::IsActive())
		{
			code += RoundingModifier<Float32Type>::OpCodeModifier();
		}
		else
		{
			code += ".approx";
		}

		return code + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float32Type);
};

template<>
class RootInstruction<Float64Type> : DispatchInherit(RootInstruction), public InstructionBase_1<Float64Type>, RoundingModifier<Float64Type, true>
{
public:
	RootInstruction(const Register<Float64Type> *destination, const TypedOperand<Float64Type> *source, Float64Type::RoundingMode roundingMode) : InstructionBase_1<Float64Type>(destination, source), RoundingModifier<Float64Type, true>(roundingMode) {}

	static std::string Mnemonic() { return "sqrt"; }

	std::string OpCode() const override
	{
		return Mnemonic() + RoundingModifier<Float64Type, true>::OpCodeModifier() + Float64Type::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float64Type);
};

DispatchImplementation(RootInstruction)

}
