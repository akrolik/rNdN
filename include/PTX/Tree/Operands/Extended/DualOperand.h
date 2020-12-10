#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class DualOperand : public Operand
{
public:
	DualOperand(Operand *operandP, Operand *operandQ) : m_operandP(operandP), m_operandQ(operandQ) {}

	// Properties

	const Operand *GetOperandP() const { return m_operandP; }
	Operand *GetOperandP() { return m_operandP; }
	void SetOperandP(Operand *operandP) { m_operandP = operandP; }

	const Operand *GetOperandQ() const { return m_operandQ; }
	Operand *GetOperandQ() { return m_operandQ; }
	void SetOperandQ(Operand *operandQ) { m_operandQ = operandQ; }

	// Formatting

	std::string ToString() const override
	{
		return m_operandP->ToString() + "|" + m_operandQ->ToString();
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX:DualOperand";
		j["operand_p"] = m_operandP->ToJSON();
		j["operand_q"] = m_operandQ->ToJSON();
		return j;
	}

private:
	Operand *m_operandP = nullptr;
	Operand *m_operandQ = nullptr;
};

}
