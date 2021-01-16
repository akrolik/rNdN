#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(BitReverseInstruction)

template<class T, bool Assert = true>
class BitReverseInstruction : DispatchInherit(BitReverseInstruction), public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(BitReverseInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase;

	// Formatting

	static std::string Mnemonic() { return "brev"; }

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

DispatchImplementation(BitReverseInstruction)

}
