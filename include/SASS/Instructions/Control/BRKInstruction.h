#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

namespace SASS {

class BRKInstruction : public PredicatedInstruction
{
public:
	// Formatting

	std::string OpCode() const override { return "BRK"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe34000000000000f;
	}
};

}
