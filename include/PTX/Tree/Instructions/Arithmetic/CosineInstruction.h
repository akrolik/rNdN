#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

DispatchInterface(CosineInstruction)

template<class T = Float32Type, bool Assert = true>
class CosineInstruction : DispatchInherit(CosineInstruction), public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(CosineInstruction,
		REQUIRE_EXACT(T, Float32Type)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	// Formatting

	static std::string Mnemonic() { return "cos"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".approx" + FlushSubnormalModifier<T>::GetOpCodeModifier() + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor &visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

}
