#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(Exp2Instruction)

template<class T = Float32Type, bool Assert = true>
class Exp2Instruction : DispatchInherit(Exp2Instruction), public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(Exp2Instruction,
		REQUIRE_EXACT(T, Float32Type)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	// Formatting

	static std::string Mnemonic() { return "ex2"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".approx" + FlushSubnormalModifier<T>::GetOpCodeModifier() + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(Exp2Instruction)

}
