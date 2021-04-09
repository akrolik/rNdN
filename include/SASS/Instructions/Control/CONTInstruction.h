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

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::x32; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
};

}
