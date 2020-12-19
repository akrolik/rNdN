#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

namespace PTX {

class DevInstruction : public PredicatedInstruction
{
public:
	DevInstruction(const std::string& instruction) : m_instruction(instruction) {}

	// Properties

	const std::string& GetInstruction() const { return m_instruction; }
	void SetInstruction(const std::string& instruction) { m_instruction = instruction; }

	// OpCode

	std::string GetOpCode() const override
	{
		return m_instruction;
	}

	// Operands

	std::vector<const Operand *> GetOperands() const override
	{
		return {};
	}

	std::vector<Operand *> GetOperands() override
	{
		return {};
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	std::string m_instruction;
};

}
