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
	IfStatement(Operand *condition, BlockStatement *trueBlock, BlockStatement *elseBlock = nullptr) : m_condition(condition), m_trueBlock(trueBlock), m_elseBlock(elseBlock) {}

	Operand *GetCondition() const { return m_condition; }
	void SetCondition(Operand *condition) { m_condition = condition; }

	BlockStatement *GetTrueBlock() const { return m_trueBlock; }
	void SetTrueBlock(BlockStatement *block) { m_trueBlock = block; }

	bool HasElseBranch() const { return m_elseBlock != nullptr; }

	BlockStatement *GetElseBlock() const { return m_elseBlock; }
	void SetElseBlock(BlockStatement *block) { m_elseBlock = block; }

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
