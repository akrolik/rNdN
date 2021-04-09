#pragma once

#include "SASS/Instructions/Instruction.h"

#include "Utils/Format.h"

namespace SASS {

class SCHIInstruction : public Instruction
{
public:
	SCHIInstruction(std::uint64_t schedule) : m_schedule(schedule) {}

	// Formatting

	std::string OpCode() const override { return ""; }
	std::string ToString() const override { return ""; }

	std::uint64_t BinaryOpCode() const override { return 0; }
	std::uint64_t ToBinary() const override
	{
		return m_schedule;
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::Schedule; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }

private:
	std::uint64_t m_schedule = 0;
};

}
