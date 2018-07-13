#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class CosineInstruction : public InstructionBase_1<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase_1<Float32Type>::InstructionBase_1;

	static std::string Mnemonic() { return "cos"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".approx" + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
	}
};

}
