#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

namespace PTX {

class DevInstruction : public PredicatedInstruction
{
public:
	DevInstruction(const std::string& instruction) : m_instruction(instruction) {}

	const std::string& GetInstruction() const { return m_instruction; }
	void SetInstruction(const std::string& instruction) { m_instruction = instruction; }

	std::string OpCode() const override
	{
		return m_instruction;
	}

	std::vector<const Operand *> Operands() const override
	{
		return {};
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	std::string m_instruction;
};

}
