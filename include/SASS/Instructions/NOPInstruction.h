#pragma once

#include "SASS/Instructions/Instruction.h"

namespace SASS {

class NOPInstruction : public Instruction
{
public:
	std::string OpCode() const override
	{
		return "NOP";
	}
};

}
