#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

namespace SASS {

class NOPInstruction : public PredicatedInstruction
{
public:
	// Formatting

	std::string OpCode() const override { return "NOP"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x50b0000000000f00;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
};

}
