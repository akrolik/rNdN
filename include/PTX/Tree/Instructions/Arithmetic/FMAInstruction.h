#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	FMAInstruction(const Register<T> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, const TypedOperand<T> *sourceC, typename T::RoundingMode roundingMode) : InstructionBase_3<T>(destination, sourceA, sourceB, sourceC), RoundingModifier<T, true>(roundingMode) {}

	static std::string Mnemonic() { return "fma"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + RoundingModifier<T, true>::OpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		if constexpr(SaturateModifier<T>::Enabled)
		{
			code += SaturateModifier<T>::OpCodeModifier();
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(FMAInstruction)

}
