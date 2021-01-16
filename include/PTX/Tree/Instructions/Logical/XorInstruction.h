#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(XorInstruction)

template<class T, bool Assert = true>
class XorInstruction : DispatchInherit(XorInstruction), public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(XorInstruction,
		REQUIRE_EXACT(T, 
			PredicateType, Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	// Formatting

	static std::string Mnemonic() { return "xor"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

}
