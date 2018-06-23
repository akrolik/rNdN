#pragma once

#include <vector>

#include "PTX/Operands/Operand.h"


namespace PTX {

class ListOperand : public Operand
{
public:
	ListOperand() {}
	ListOperand(const std::vector<const Operand *>& operands) : m_operands(operands) {}

	void AddOperand(const Operand *operand)
	{
		m_operands.push_back(operand);
	}

	std::string ToString() const override
	{
		std::string code = "(";
		bool first = true;
		for (const auto& operand : m_operands)
		{
			if (!first)
			{
				code += ", ";
			}
			first = false;
			code += operand->ToString();
		}
		return code + ")";
	}

private:
	std::vector<const Operand *> m_operands;
};

}
