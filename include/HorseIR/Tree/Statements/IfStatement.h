#pragma once

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Operand.h"
#include "HorseIR/Tree/Statements/BlockStatement.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class IfStatement : public Statement
{
public:
	IfStatement(Operand *condition, BlockStatement *trueBlock, BlockStatement *elseBlock = nullptr, int line = 0) : Statement(line), m_condition(condition), m_trueBlock(trueBlock), m_elseBlock(elseBlock) {}

	IfStatement *Clone() const override
	{
		return new IfStatement(m_condition->Clone(), m_trueBlock->Clone(), (m_elseBlock == nullptr) ? nullptr : m_elseBlock->Clone());
	}

	// Condition

	const Operand *GetCondition() const { return m_condition; }
	Operand *GetCondition() { return m_condition; }

	void SetCondition(Operand *condition) { m_condition = condition; }

	// True branch

	const BlockStatement *GetTrueBlock() const { return m_trueBlock; }
	BlockStatement *GetTrueBlock() { return m_trueBlock; }

	void SetTrueBlock(BlockStatement *block) { m_trueBlock = block; }

	// Else branch

	bool HasElseBranch() const { return m_elseBlock != nullptr; }

	const BlockStatement *GetElseBlock() const { return m_elseBlock; }
	BlockStatement *GetElseBlock() { return m_elseBlock; }

	void SetElseBlock(BlockStatement *block) { m_elseBlock = block; }

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			m_trueBlock->Accept(visitor);
			if (m_elseBlock != nullptr)
			{
				m_elseBlock->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			m_trueBlock->Accept(visitor);
			if (m_elseBlock != nullptr)
			{
				m_elseBlock->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	Operand *m_condition = nullptr;
	BlockStatement *m_trueBlock = nullptr;
	BlockStatement *m_elseBlock = nullptr;
};

}
