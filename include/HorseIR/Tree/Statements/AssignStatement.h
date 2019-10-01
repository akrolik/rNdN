#pragma once

#include <vector>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/LValue.h"
#include "HorseIR/Tree/Expressions/Expression.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class AssignStatement : public Statement
{
public:
	AssignStatement(const std::vector<LValue *>& targets, Expression *expression) : m_targets(targets), m_expression(expression) {}

	AssignStatement *Clone() const override
	{
		std::vector<LValue *> targets;
		for (const auto& target : m_targets)
		{
			targets.push_back(target->Clone());
		}
		return new AssignStatement(targets, m_expression->Clone());
	}

	const std::vector<LValue *>& GetTargets() const { return m_targets; }
	unsigned int GetTargetCount() const { return m_targets.size(); }
	void SetTargets(const std::vector<LValue *>& targets) { m_targets = targets; }

	Expression *GetExpression() const { return m_expression; }
	void SetExpression(Expression *expression) { m_expression = expression; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& target : m_targets)
			{
				target->Accept(visitor);
			}
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& target : m_targets)
			{
				target->Accept(visitor);
			}
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	std::vector<LValue *> m_targets;
	Expression *m_expression = nullptr;
};

}
