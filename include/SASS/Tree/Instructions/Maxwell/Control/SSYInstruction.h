#pragma once

#include "SASS/Tree/Instructions/Maxwell/Control/DivergenceInstruction.h"

namespace SASS {
namespace Maxwell {

class SSYInstruction : public DivergenceInstruction
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

	std::string OpCode() const override { return "SSY"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe290000000000000;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
};

}
}
