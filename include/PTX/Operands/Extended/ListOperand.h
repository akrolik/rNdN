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

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::ListOperand";
		for (const auto& operand : m_operands)
		{
			j["operands"].push_back(operand->ToJSON());
		}
		return j;
	}

private:
	std::vector<const Operand *> m_operands;
};

}
