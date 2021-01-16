#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(MADWideInstruction)

template<class T, bool Assert = true>
class MADWideInstruction : DispatchInherit(MADWideInstruction), public InstructionBase_3<typename T::WideType, T, T, typename T::WideType>
{
public:
	REQUIRE_TYPE_PARAM(MADWideInstruction,
		REQUIRE_EXACT(T, Int16Type, Int32Type, UInt16Type, UInt32Type)
	);

	using InstructionBase_3<typename T::WideType, T, T, typename T::WideType>::InstructionBase_3;

	// Formatting

	static std::string Mnemonic() { return "mad"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".wide" + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

}
