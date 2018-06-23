#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Statements/Label.h"

namespace PTX {

class BranchInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchInstruction(const Label *label, bool uniform = false) : UniformModifier(uniform), m_label(label) {}

	std::string OpCode() const override
	{
		return "bra" + UniformModifier::OpCodeModifier();
	}

	std::string Operands() const override
	{
		return m_label->ToString();
	}

private:
	const Label *m_label = nullptr;
};

}
