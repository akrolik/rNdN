#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

namespace SASS {

class EXITInstruction : public PredicatedInstruction
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

	std::string OpCode() const override { return "EXIT"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe30000000000000f;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Control; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
};

}
