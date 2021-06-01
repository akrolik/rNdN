#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(ActiveMaskInstruction)

template<class T, bool Assert = true>
class ActiveMaskInstruction : DispatchInherit(ActiveMaskInstruction), public InstructionBase_0<T>
{
public:
	REQUIRE_TYPE_PARAM(ActiveMaskInstruction,
		REQUIRE_EXACT(T, Bit32Type)
	);

	using InstructionBase_0<T>::InstructionBase_0;

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Formatting

	static std::string Mnemonic() { return "activemask"; }

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
