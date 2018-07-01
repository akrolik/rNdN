#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

class ExitInstruction : public PredicatedInstruction
{
public:
	ExitInstruction() {}

	std::string OpCode() const override
	{
		return "exit";
	}

	std::vector<const Operand *> Operands() const override
	{
		return {};
	}
};

}
