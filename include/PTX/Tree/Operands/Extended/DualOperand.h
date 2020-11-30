#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class DualOperand : public Operand
{
public:
	DualOperand(const Operand *operandP, const Operand *operandQ) : m_operandP(operandP), m_operandQ(operandQ) {}

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
	const Operand *m_operandP = nullptr;
	const Operand *m_operandQ = nullptr;

};

}
