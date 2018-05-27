#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

class ExitInstruction : public InstructionStatement
{
public:
	ExitInstruction() {}

	std::string OpCode() const
	{
		return "exit";
	}

	std::string Operands() const
	{
		return "";
	}
};

}
