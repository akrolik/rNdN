#pragma once

#include "PTX/Tree/Statements/Statement.h"

#include "PTX/Tree/Operands/Label.h"

namespace PTX {

class LabelStatement : public Statement
{
public:
	LabelStatement(Label *label) : m_label(label) {}

	// Properties

	const Label *GetLabel() const { return m_label; }
	Label *GetLabel() { return m_label; }
	void SetLabel(Label *label) { m_label = label; }

	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::LabelStatement";
		j["label"] = m_label->ToJSON();
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_label->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_label->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

private:
	Label *m_label = nullptr;
};

}
