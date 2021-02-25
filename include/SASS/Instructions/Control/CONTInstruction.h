#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

namespace SASS {

class CONTInstruction : public PredicatedInstruction
{
public:
	// Formatting

	std::string OpCode() const override { return "CONT"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe30000000000000f;
	}
};

}
