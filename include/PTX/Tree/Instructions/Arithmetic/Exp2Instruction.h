#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class Exp2Instruction : public InstructionBase_1<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase_1<Float32Type>::InstructionBase_1;

	static std::string Mnemonic() { return "ex2"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".approx" + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
	}
};

}
