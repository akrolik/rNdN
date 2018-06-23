#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

namespace PTX {

class ReturnInstruction : public InstructionStatement, public UniformModifier
{
public:
	ReturnInstruction(bool uniform = false) : UniformModifier(uniform) {}

	std::string OpCode() const override
	{
		return "ret" + UniformModifier::OpCodeModifier();
	}

	std::vector<const Operand *> Operands() const override
	{
		return {};
	}
};

}
