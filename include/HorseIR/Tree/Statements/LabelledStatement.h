#pragma once

#include <string>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class LabelledStatement : public Statement
{
public:
	LabelledStatement(const std::string& name, Statement *statement) : m_labelName(name), m_statement(statement) {}

	const std::string& GetLabelName() const { return m_labelName; }
	void SetLabelName(const std::string& name) { m_labelName = name; }

	Statement *GetStatement() const { return m_statement; }
	void SetStatement(Statement *statement) { m_statement = statement; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_statement->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_statement->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	std::string m_labelName;
	Statement *m_statement;
};

}
