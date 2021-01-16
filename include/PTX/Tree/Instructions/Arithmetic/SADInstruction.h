#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(SADInstruction)

template<class T, bool Assert = true>
class SADInstruction : DispatchInherit(SADInstruction), public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(SADInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	// Formatting

	static std::string Mnemonic() { return "sad"; }

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

DispatchImplementation(SADInstruction)

}
