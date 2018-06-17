#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

class ExitInstruction : public InstructionStatement
{
public:
	ExitInstruction() {}

	std::string OpCode() const override
	{
		return "exit";
	}

	std::string Operands() const override
	{
		return "";
	}
};

}
