#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Statements/Label.h"

namespace PTX {

class BranchInstruction : public PredicatedInstruction
{
public:
	BranchInstruction(Label *label, bool uniform = false) : m_label(label), m_uniform(uniform) {}

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
	bool m_uniform = false;
};

}
