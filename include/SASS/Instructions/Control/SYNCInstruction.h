#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

namespace SASS {

class SYNCInstruction : public PredicatedInstruction
{
public:
	// Formatting

	std::string OpCode() const override { return "SYNC"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xf0f800000000000f;
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::x32; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
};

}
