#pragma once

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Operand.h"
#include "HorseIR/Tree/Statements/BlockStatement.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class WhileStatement : public Statement
{
public:
	WhileStatement(Operand *condition, BlockStatement *body) : m_condition(condition), m_body(body) {}

	WhileStatement *Clone() const override
	{
		return new WhileStatement(m_condition->Clone(), m_body->Clone());
	}

	// Condition

	const Operand *GetCondition() const { return m_condition; }
	Operand *GetCondition() { return m_condition; }

	void SetCondition(Operand *condition) { m_condition = condition; }
	
	// Body

	const BlockStatement *GetBody() const { return m_body; }
	BlockStatement *GetBody() { return m_body; }

	void SetBody(BlockStatement *body) { m_body = body; }

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			m_body->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			m_body->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	Operand *m_condition = nullptr;
	BlockStatement *m_body = nullptr;
};

}
