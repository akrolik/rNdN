#pragma once

#include "PTX/Operands/Operand.h"


namespace PTX {

class InvertedOperand : public Operand
{
public:
	InvertedOperand(const Operand *operand) : m_operand(operand) {}

	std::string ToString() const override
	{
		return "!" + m_operand->ToString();
	}

private:
	const Operand *m_operand = nullptr;

};

}
