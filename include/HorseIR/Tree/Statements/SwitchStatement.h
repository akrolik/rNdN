#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/Expressions/Identifier.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class SwitchStatement : public Statement
{
public:
	SwitchStatement(const std::vector<Identifier *>& conditions, const std::vector<std::string>& labels) : m_conditions(conditions), m_labels(labels) {}

	const std::vector<Identifier *>& GetConditions() const { return m_conditions; }
	void SetConditions(const std::vector<Identifier *>& conditions) { m_conditions = conditions; }

	const std::vector<std::string>& GetLabels() const { return m_labels; }
	void SetLabels(const std::vector<std::string>& labels) { m_labels = labels; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& condition : m_conditions)
			{
				condition->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& condition : m_conditions)
			{
				condition->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	std::vector<Identifier *> m_conditions;
	std::vector<std::string> m_labels;
};

}
