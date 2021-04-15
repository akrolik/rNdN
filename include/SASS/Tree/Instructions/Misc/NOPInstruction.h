#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

namespace SASS {

class NOPInstruction : public PredicatedInstruction
{
public:
	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { };
	}

	// Formatting

	std::string OpCode() const override { return "NOP"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x50b0000000000f00;
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::x32; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
};

}
