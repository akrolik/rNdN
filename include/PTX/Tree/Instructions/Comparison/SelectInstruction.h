#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(SelectInstruction)

template<class T, bool Assert = true>
class SelectInstruction : DispatchInherit(SelectInstruction), public InstructionBase_3<T, T, T, PredicateType>
{
public:
	REQUIRE_TYPE_PARAM(SelectInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_3<T, T, T, PredicateType>::InstructionBase_3;

	// Formatting

	static std::string Mnemonic() { return "selp"; }

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
