#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(MultiplyWideInstruction)

template<class T, bool Assert = true>
class MultiplyWideInstruction : DispatchInherit(MultiplyWideInstruction), public InstructionBase_2<typename T::WideType, T>
{
public:
	REQUIRE_TYPE_PARAM(MultiplyWideInstruction,
		REQUIRE_EXACT(T, Int16Type, Int32Type, UInt16Type, UInt32Type)
	);

	using InstructionBase_2<typename T::WideType, T>::InstructionBase_2;

	static std::string Mnemonic() { return "mul"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".wide" + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(MultiplyWideInstruction)
 
}
