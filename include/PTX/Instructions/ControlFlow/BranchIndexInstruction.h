#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Statements/Label.h"

namespace PTX {

class BranchIndexInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchIndexInstruction(const Register<UInt32Type> *index, const std::vector<Label *>& labels, bool uniform = false) : UniformModifier(uniform), m_index(index), m_labels(labels) {}

	std::string OpCode() const override
	{
		if (m_uniform)
		{
			return "brx.idx.uni";
		}
		return "brx.idx";
	}

	std::string Operands() const override
	{
		std::string code = m_index->ToString();
		for (auto it = m_labels.cbegin(); it != m_labels.cend(); ++it)
		{
			code += ", " + (*it)->ToString();
		}
		return code;
	}

private:
	const Register<UInt32Type> *m_index = nullptr;
	const std::vector<Label *> m_labels;
};

}
