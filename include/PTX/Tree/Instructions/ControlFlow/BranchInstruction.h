#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/UniformModifier.h"

#include "PTX/Tree/Operands/Label.h"

namespace PTX {

class BranchInstruction : public PredicatedInstruction, public UniformModifier
{
public:
	BranchInstruction(Label *label, bool uniform = false) : UniformModifier(uniform), m_label(label) {}
	BranchInstruction(Label *label, Register<PredicateType> *predicate, bool negate = false, bool uniform = false) : PredicatedInstruction(predicate, negate), UniformModifier(uniform), m_label(label) {}

	// Properties

	const Label *GetLabel() const { return m_label; }
	Label *GetLabel() { return m_label; }
	void SetLabel(Label *label) { m_label = label; }

	// Formatting

	static std::string Mnemonic() { return "bra"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + UniformModifier::GetOpCodeModifier();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_label };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_label };
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	Label *m_label = nullptr;
};

}
