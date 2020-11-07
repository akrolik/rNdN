#pragma once

#include "SASS/Instructions/Instruction.h"

namespace SASS {

class EXITInstruction : public Instruction
{
public:
	// Formatting

	std::string OpCode() const override { return "EXIT"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe30000000000000f;
	}
};

}
