#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(BitFieldExtractInstruction)

template<class T, bool Assert = true>
class BitFieldExtractInstruction : DispatchInherit(BitFieldExtractInstruction), public InstructionBase_3<T, T, UInt32Type, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(BitFieldExtractInstruction,
		REQUIRE_EXACT(T,
			Int32Type, Int64Type,
			UInt32Type, UInt64Type
		)
	);

	using InstructionBase_3<T, T, UInt32Type, UInt32Type>::InstructionBase;

	// Formatting

	static std::string Mnemonic() { return "bfe"; }

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
