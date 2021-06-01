#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Tree/Operands/Label.h"

namespace PTX {

class BranchIndexInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchIndexInstruction(Register<UInt32Type> *index, const std::vector<Label *>& labels, bool uniform = false) : UniformModifier(uniform), m_index(index), m_labels(labels) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const Register<UInt32Type> *GetIndex() const { return m_index; }
	Register<UInt32Type> *GetIndex() { return m_index; }
	void SetIndex(Register<UInt32Type> *index) { m_index = index; }

	std::vector<const Label *> GetLabels() const
	{
		return { std::begin(m_labels), std::end(m_labels) };
	}
	std::vector<Label *>& GetLabels() { return m_labels; }
	void SetLabels(const std::vector<Label *>& labels) { m_labels = labels; }

	// Formatting

	static std::string Mnemonic() { return "brx.idx"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + UniformModifier::GetOpCodeModifier();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands({ m_index });
		for (const auto& label : m_labels)
		{
			operands.push_back(label);
		}
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands({ m_index });
		for (auto& label : m_labels)
		{
			operands.push_back(label);
		}
		return operands;
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	Register<UInt32Type> *m_index = nullptr;
	std::vector<Label *> m_labels;
};

}
