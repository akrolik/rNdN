#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(SineInstruction)

template<class T = Float32Type, bool Assert = true>
class SineInstruction : DispatchInherit(SineInstruction), public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(SineInstruction,
		REQUIRE_EXACT(T, Float32Type)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	static std::string Mnemonic() { return "sin"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".approx" + FlushSubnormalModifier<T>::OpCodeModifier() + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(SineInstruction)

}
