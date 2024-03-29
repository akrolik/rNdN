#pragma once

#include <vector>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class ReturnStatement : public Statement
{
public:
	ReturnStatement(const std::vector<Operand *>& operands, int line = 0) : Statement(line), m_operands(operands) {}

	ReturnStatement *Clone() const override
	{
		std::vector<Operand *> operands;
		for (const auto& operand : m_operands)
		{
			operands.push_back(operand->Clone());
		}
		return new ReturnStatement(operands);
	}

	// Operands

	unsigned int GetOperandsCount() const { return m_operands.size(); }

	std::vector<const Operand *> GetOperands() const
	{
		return { std::begin(m_operands), std::end(m_operands) };
	}
	std::vector<Operand *>& GetOperands() { return m_operands; }

	void SetOperands(const std::vector<Operand *>& operands) { m_operands = operands; }

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& operand : m_operands)
			{
				operand->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& operand : m_operands)
			{
				operand->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	std::vector<Operand *> m_operands;
};

}
