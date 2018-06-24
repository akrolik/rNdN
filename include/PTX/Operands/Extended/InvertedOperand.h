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

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::InvertedOperand";
		j["operand"] = m_operand->ToJSON();
		return j;
	}

private:
	const Operand *m_operand = nullptr;

};

}
