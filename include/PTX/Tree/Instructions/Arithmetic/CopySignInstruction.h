#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(CopySignInstruction)

template<class T, bool Assert = true>
class CopySignInstruction : DispatchInherit(CopySignInstruction), public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(CopySignInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase;

	// Formatting

	static std::string Mnemonic() { return "copysign"; }
	
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
