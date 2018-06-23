#pragma once

#include "PTX/Operands/Operand.h"



namespace PTX {

class DualOperand : public Operand
{
public:
	DualOperand(const Operand *operandP, const Operand *operandQ) : m_operandP(operandP), m_operandQ(operandQ) {}

	std::string ToString() const override
	{
		return m_operandP->ToString() + "|" + m_operandQ->ToString();
	}

private:
	const Operand *m_operandP = nullptr;
	const Operand *m_operandQ = nullptr;

};

}
