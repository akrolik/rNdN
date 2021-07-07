#pragma once

#include <vector>

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class ListOperand : public Operand, public DispatchBase
{
public:
	ListOperand() {}
	ListOperand(const std::vector<Operand *>& operands) : m_operands(operands) {}

	// Properties

	std::vector<const Operand *> GetOperands() const
	{
		return { std::begin(m_operands), std::end(m_operands) };
	}
	std::vector<Operand *>& GetOperands() { return m_operands; }

	void SetOperands(const std::vector<Operand *>& operands) { m_operands = operands; }
	void AddOperand(Operand *operand)
	{
		m_operands.push_back(operand);
	}

	// Formatting

	std::string ToString() const override
	{
		std::string code = "(";
		auto first = true;
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

	void Accept(OperandVisitor& visitor) override
	{
		if (visitor.Visit(this))
		{
			for (auto& operand : m_operands)
			{
				operand->Accept(visitor);
			}
		}
	}

	void Accept(ConstOperandVisitor& visitor) const override
	{
		if (visitor.Visit(this))
		{
			for (const auto& operand : m_operands)
			{
				operand->Accept(visitor);
			}
		}
	}


private:
	std::vector<Operand *> m_operands;
};

}
