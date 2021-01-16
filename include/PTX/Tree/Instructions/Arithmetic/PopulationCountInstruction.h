#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(PopulationCountInstruction)

template<class T, bool Assert = true>
class PopulationCountInstruction : DispatchInherit(PopulationCountInstruction), public InstructionBase_1<UInt32Type, T>
{
public:
	REQUIRE_TYPE_PARAM(PopulationCountInstruction,
		REQUIRE_EXACT(T,
			Bit32Type, Bit64Type
		)
	);

	using InstructionBase_1<UInt32Type, T>::InstructionBase_1;

	// Formatting

	static std::string Mnemonic() { return "popc"; }

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
