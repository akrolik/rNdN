#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Statements/Label.h"

namespace PTX {

class BranchInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchInstruction(Label *label, bool uniform = false) : UniformModifier(uniform), m_label(label) {}

	std::string OpCode() const
	{
		if (m_uniform)
		{
			return "bra.uni";
		}
		return "bra";
	}

	std::string Operands() const
	{
		return m_label->ToString();
	}

private:
	Label *m_label = nullptr;
};

}
