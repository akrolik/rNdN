#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(ShiftRightInstruction)

template<class T, bool Assert = true>
class ShiftRightInstruction : DispatchInherit(ShiftRightInstruction), public InstructionBase_2<T, T, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(ShiftRightInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	static std::string Mnemonic() { return "shr"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitor

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T)
};

DispatchImplementation(ShiftRightInstruction)

}
