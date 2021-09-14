#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(TanhInstruction)

template<class T = Float32Type, bool Assert = true>
class TanhInstruction : DispatchInherit(TanhInstruction), public InstructionBase_1<T>
{
public:
	REQUIRE_TYPE_PARAM(TanhInstruction,
		REQUIRE_EXACT(T, Float16Type, Float16x2Type, Float32Type)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Formatting

	static std::string Mnemonic() { return "tanh"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".approx" + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

}
