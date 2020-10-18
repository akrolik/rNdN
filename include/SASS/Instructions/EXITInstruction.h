#pragma once

#include "SASS/Instructions/Instruction.h"

namespace SASS {

class EXITInstruction : public Instruction
{
public:
	std::string OpCode() const override { return "EXIT"; }
};

}