#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(BitFieldInsertInstruction)

template<class T, bool Assert = true>
class BitFieldInsertInstruction : DispatchInherit(BitFieldInsertInstruction), public InstructionBase_4<T, T, T, UInt32Type, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(BitFieldInsertInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type
		)
	);

	using InstructionBase_4<T, T, T, UInt32Type, UInt32Type>::InstructionBase;

	// Analysis properties

	bool HasSideEffect() const override { return false; }
  	
	// Formatting

	static std::string Mnemonic() { return "bfi"; }

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
