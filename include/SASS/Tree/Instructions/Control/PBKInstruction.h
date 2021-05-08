#pragma once

#include "SASS/Tree/Instructions/Control/DivergenceInstruction.h"

namespace SASS {

class PBKInstruction : public DivergenceInstruction
{
public:
	using DivergenceInstruction::DivergenceInstruction;

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

	std::string OpCode() const override { return "PBK"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe2a0000000000000;
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::Control; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
};

}
