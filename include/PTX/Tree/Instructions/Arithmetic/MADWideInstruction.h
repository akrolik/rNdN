#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	static std::string Mnemonic() { return "mad"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".wide" + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(MADWideInstruction)

}
