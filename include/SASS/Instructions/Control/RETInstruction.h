#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

namespace SASS {

class RETInstruction : public PredicatedInstruction
{
public:
	// Formatting

	std::string OpCode() const override { return "RET"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe32000000000000f;
	}
};

}
