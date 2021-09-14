#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

namespace PTX {

class BreakpointInstruction : public PredicatedInstruction, public DispatchBase
{
public:
	using PredicatedInstruction::PredicatedInstruction;

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Formatting

	static std::string Mnemonic() { return "brkpt"; }

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

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }
};

}
