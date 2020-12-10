#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class InvertedOperand : public Operand
{
public:
	InvertedOperand(Operand *operand) : m_operand(operand) {}

	// Properties

	const Operand *GetOperand() const { return m_operand; }
	Operand *GetOperand() { return m_operand; }
	void SetOperand(Operand *operand) { m_operand = operand; }

	// Formatting

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
	Operand *m_operand = nullptr;
};

}
