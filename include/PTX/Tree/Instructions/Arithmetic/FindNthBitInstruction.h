#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(FindNthBitInstruction)

template<class T, bool Assert = true>
class FindNthBitInstruction : DispatchInherit(FindNthBitInstruction), public InstructionBase_3<Bit32Type, Bit32Type, T, Int32Type>
{
public:
	REQUIRE_TYPE_PARAM(FindNthBitInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Int32Type, UInt32Type
		)
	);

	using InstructionBase_3<Bit32Type, Bit32Type, T, Int32Type>::InstructionBase_3;

	// Formatting

	static std::string Mnemonic() { return "fns"; }

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

DispatchImplementation(FindNthBitInstruction)

}
