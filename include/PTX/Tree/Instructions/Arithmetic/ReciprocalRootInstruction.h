#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

DispatchInterface(ReciprocalRootInstruction)

template<class T, bool Assert = true>
class ReciprocalRootInstruction : DispatchInherit(ReciprocalRootInstruction), public InstructionBase_1<T>, public FlushSubnormalModifier<T, true>
{
public:
	REQUIRE_TYPE_PARAM(ReciprocalRootInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Formatting

	static std::string Mnemonic() { return "rsqrt"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".approx" + FlushSubnormalModifier<T, true>::GetOpCodeModifier() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

}
