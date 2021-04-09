#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

namespace SASS {

class EXITInstruction : public PredicatedInstruction
{
public:
	// Formatting

	std::string OpCode() const override { return "EXIT"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe30000000000000f;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
};

}
