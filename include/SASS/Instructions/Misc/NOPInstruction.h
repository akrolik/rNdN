#pragma once

#include "SASS/Instructions/Instruction.h"

namespace SASS {

class NOPInstruction : public Instruction
{
public:
	// Formatting

	std::string OpCode() const override { return "NOP"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x50b0000000000f00;
	}
};

}
