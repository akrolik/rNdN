#pragma once

#include <vector>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class IfStatement : public Statement
{
public:
	IfStatement(Operand *condition, const std::vector<Statement *> trueStatements, const std::vector<Statement *> falseStatements) : m_condition(condition), m_trueStatements(trueStatements), m_falseStatements(falseStatements) {}

	Operand *GetCondition() const { return m_condition; }
	void SetCondition(Operand *condition) { m_condition = condition; }

	const std::vector<Statement *>& GetTrueStatements() const { return m_trueStatements; }
	void SetTrueStatements(const std::vector<Statement *>& statements) { m_trueStatements = statements; }

	bool HasFalseBranch() const { return m_falseStatements.size() > 0; }

	const std::vector<Statement *>& GetFalseStatements() const { return m_falseStatements; }
	void SetFalseStatements(const std::vector<Statement *>& statements) { m_falseStatements = statements; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			for (auto& statement : m_trueStatements)
			{
				statement->Accept(visitor);
			}
			for (auto& statement : m_falseStatements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_condition->Accept(visitor);
			for (const auto& statement : m_trueStatements)
			{
				statement->Accept(visitor);
			}
			for (const auto& statement : m_falseStatements)
			{
				statement->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	Operand *m_condition = nullptr;
	std::vector<Statement *> m_trueStatements;
	std::vector<Statement *> m_falseStatements;
};

}
