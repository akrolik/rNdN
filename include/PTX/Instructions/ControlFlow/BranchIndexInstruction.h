#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Statements/Label.h"

namespace PTX {

class BranchIndexInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchIndexInstruction(Register<UInt32Type> *index, std::vector<Label *> labels, bool uniform = false) : UniformModifier(uniform), m_index(index), m_labels(labels) {}

	std::string OpCode() const
	{
		if (m_uniform)
		{
			return "brx.idx.uni";
		}
		return "brx.idx";
	}

	std::string Operands() const
	{
		std::string code = m_index->ToString();
		for (auto it = m_labels.cbegin(); it != m_labels.cend(); ++it)
		{
			code += ", " + (*it)->ToString();
		}
		return code;
	}

private:
	Register<UInt32Type> *m_index = nullptr;
	std::vector<Label *> m_labels;
};

}
