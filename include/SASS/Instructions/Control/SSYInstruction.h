#pragma once

#include "SASS/Instructions/Control/DivergenceInstruction.h"

namespace SASS {

class SSYInstruction : public DivergenceInstruction
{
public:
	using DivergenceInstruction::DivergenceInstruction;

	// Formatting

	std::string OpCode() const override { return "SSY"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe290000000000000;
	}
};

}
