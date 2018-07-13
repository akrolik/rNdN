#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Statements/Label.h"

namespace PTX {

class BranchInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchInstruction(const Label *label, bool uniform = false) : UniformModifier(uniform), m_label(label) {}

	static std::string Mnemonic() { return "bra"; }

	std::string OpCode() const override
	{
		return Mnemonic() + UniformModifier::OpCodeModifier();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_label };
	}

private:
	const Label *m_label = nullptr;
};

}
