#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(NotInstruction)

template<class T, bool Assert = true>
class NotInstruction : DispatchInherit(NotInstruction), public InstructionBase_1<T>
{
public:
	REQUIRE_TYPE_PARAM(NotInstruction,
		REQUIRE_EXACT(T,
			PredicateType, Bit16Type, Bit32Type, Bit64Type
		)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	// Formatting

	static std::string Mnemonic() { return "not"; }

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
