#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(ShiftLeftInstruction)

template<class T, bool Assert = true>
class ShiftLeftInstruction : DispatchInherit(ShiftLeftInstruction), public InstructionBase_2<T, T, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(ShiftLeftInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Formatting

	static std::string Mnemonic() { return "shl"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T)
};

}
