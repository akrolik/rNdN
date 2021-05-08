#pragma once

#include "SASS/Tree/Instructions/Instruction.h"

#include "Utils/Format.h"

namespace SASS {

class SCHIInstruction : public Instruction
{
public:
	SCHIInstruction(std::uint64_t schedule) : m_schedule(schedule) {}

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

	std::string OpCode() const override { return ""; }

	std::uint64_t BinaryOpCode() const override { return 0; }
	std::uint64_t ToBinary() const override
	{
		return m_schedule;
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::SCHI; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	std::uint64_t m_schedule = 0;
};

}
