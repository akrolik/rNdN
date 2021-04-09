#pragma once

#include "SASS/Instructions/Control/DivergenceInstruction.h"

namespace SASS {

class PCNTInstruction : public DivergenceInstruction
{
public:
	using DivergenceInstruction::DivergenceInstruction;

	// Formatting

	std::string OpCode() const override { return "PCNT"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe2b0000000000000;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
};

}
