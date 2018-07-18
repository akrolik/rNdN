#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

class DevInstruction : public PredicatedInstruction
{
public:
	DevInstruction(const std::string& instruction) : m_instruction(instruction) {}

	std::string OpCode() const override
	{
		return m_instruction;
	}

	std::string Terminator() const override
	{
		return "";
	}

	std::vector<const Operand *> Operands() const override
	{
		return {};
	}

private:
	std::string m_instruction;
};

}
