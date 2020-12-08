#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(ActiveMaskInstruction)

template<class T>
class ActiveMaskInstruction : DispatchInherit(ActiveMaskInstruction), public InstructionBase_0<T>
{
public:
	REQUIRE_TYPE_PARAM(ActiveMaskInstruction,
		REQUIRE_EXACT(T, Bit32Type)
	);

	using InstructionBase_0<T>::InstructionBase_0;

	static std::string Mnemonic() { return "activemask"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(ActiveMaskInstruction)

}
