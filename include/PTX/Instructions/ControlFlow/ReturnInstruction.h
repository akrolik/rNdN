#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

namespace PTX {

class ReturnInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	ReturnInstruction(bool uniform = false) : UniformModifier(uniform) {}

	static std::string Mnemonic() { return "ret"; }

	std::string OpCode() const override
	{
		return Mnemonic() + UniformModifier::OpCodeModifier();
	}

	std::vector<const Operand *> Operands() const override
	{
		return {};
	}
};

}
