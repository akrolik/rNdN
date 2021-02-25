#pragma once

#include "SASS/Instructions/Control/DivergenceInstruction.h"

namespace SASS {

class PBKInstruction : public DivergenceInstruction
{
public:
	using DivergenceInstruction::DivergenceInstruction;

	// Formatting

	std::string OpCode() const override { return "PBK"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe2a0000000000000;
	}
};

}
