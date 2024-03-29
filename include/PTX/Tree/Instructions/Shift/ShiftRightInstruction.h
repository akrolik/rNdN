#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(ShiftRightInstruction)

template<class T, bool Assert = true>
class ShiftRightInstruction : DispatchInherit(ShiftRightInstruction), public InstructionBase_2<T, T, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(ShiftRightInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Int16Type, Int32Type, Int64Type
		)
	);

	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Formatting

	static std::string Mnemonic() { return "shr"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitor

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T)
};

}
