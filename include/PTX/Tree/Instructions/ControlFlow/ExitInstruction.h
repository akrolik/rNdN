#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

namespace PTX {

class ExitInstruction : public PredicatedInstruction
{
public:
	ExitInstruction() {}

	static std::string Mnemonic() { return "exit"; }

	std::string OpCode() const override
	{
		return Mnemonic();
	}

	std::vector<const Operand *> Operands() const override
	{
		return {};
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }
};

}
