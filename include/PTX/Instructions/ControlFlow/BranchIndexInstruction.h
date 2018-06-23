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
		return "brx.idx" + UniformModifier::OpCodeModifier();
	}

	std::vector<const Operand *> Operands() const override
	{
		std::vector<const Operand *> operands({ m_index });
		for (const auto& label : m_labels)
		{
			operands.push_back(label);
		}
		return operands;
	}

private:
	const Register<UInt32Type> *m_index = nullptr;
	const std::vector<Label *> m_labels;
};

}
