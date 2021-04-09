#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/UniformModifier.h"

namespace PTX {

class ReturnInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	ReturnInstruction(bool uniform = false) : UniformModifier(uniform) {}

	// formatting

	static std::string Mnemonic() { return "ret"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + UniformModifier::GetOpCodeModifier();
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

	void Accept(HierarchicalVisitor& visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }
};

}
