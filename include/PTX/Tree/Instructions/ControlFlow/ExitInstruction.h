#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

namespace PTX {

class ExitInstruction : public PredicatedInstruction
{
public:
	ExitInstruction() {}

	// Formatting

	static std::string Mnemonic() { return "exit"; }

	std::string GetOpCode() const override
	{
		return Mnemonic();
	}

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
};

}
