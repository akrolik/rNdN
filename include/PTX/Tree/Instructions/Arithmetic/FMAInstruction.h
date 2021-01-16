#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(FMAInstruction)

template<class T, bool Assert = true>
class FMAInstruction : DispatchInherit(FMAInstruction), public InstructionBase_3<T>, public RoundingModifier<T, true>, public FlushSubnormalModifier<T>, public SaturateModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(FMAInstruction,
		REQUIRE_EXACT(T,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	FMAInstruction(Register<T> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, TypedOperand<T> *sourceC, typename T::RoundingMode roundingMode)
		: InstructionBase_3<T>(destination, sourceA, sourceB, sourceC), RoundingModifier<T, true>(roundingMode) {}

	// Formatting

	static std::string Mnemonic() { return "fma"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic() + RoundingModifier<T, true>::GetOpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::GetOpCodeModifier();
		}
		if constexpr(SaturateModifier<T>::Enabled)
		{
			code += SaturateModifier<T>::GetOpCodeModifier();
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(FMAInstruction)

}
