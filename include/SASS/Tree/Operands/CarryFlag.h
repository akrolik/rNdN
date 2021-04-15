#pragma once

#include "SASS/Tree/Operands/Operand.h"

namespace SASS {

class CarryFlag : public Operand
{
public:
	// Formatting

	std::string ToString() const override
	{
		return "CC";
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		return 0x0;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
};

static Operand *CC = new CarryFlag();

}
