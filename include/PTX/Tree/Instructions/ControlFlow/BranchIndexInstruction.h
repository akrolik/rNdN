#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Tree/Statements/Label.h"

namespace PTX {

class BranchIndexInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchIndexInstruction(const Register<UInt32Type> *index, const std::vector<const Label *>& labels, bool uniform = false) : UniformModifier(uniform), m_index(index), m_labels(labels) {}

	static std::string Mnemonic() { return "brx.idx"; }

	const Register<UInt32Type> *GetIndex() const { return m_index; }
	void SetIndex(const Register<UInt32Type> *index) { m_index = index; }

	const std::vector<const Label *>& GetLabels() const { return m_labels; }
	void SetLabels(const std::vector<const Label *>& labels) { m_labels = labels; }

	std::string OpCode() const override
	{
		return Mnemonic() + UniformModifier::OpCodeModifier();
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

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	const Register<UInt32Type> *m_index = nullptr;
	std::vector<const Label *> m_labels;
};

}
