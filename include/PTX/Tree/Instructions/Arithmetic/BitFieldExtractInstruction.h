#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	static std::string Mnemonic() { return "bfe"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(BitFieldExtractInstruction)
 
}
