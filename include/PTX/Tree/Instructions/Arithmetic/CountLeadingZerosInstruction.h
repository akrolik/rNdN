#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(CountLeadingZerosInstruction)

template<class T, bool Assert = true>
class CountLeadingZerosInstruction : DispatchInherit(CountLeadingZerosInstruction), public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE_PARAM(CountLeadingZerosInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type
		)
	);

	using InstructionBase_1<UInt32Type, T>::InstructionBase_1;

	// Formatting

	static std::string Mnemonic() { return "clz"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(CountLeadingZerosInstruction)

}
