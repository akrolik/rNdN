#pragma once

#include "PTX/Tree/Statements/Statement.h"

#include "PTX/Tree/Declarations/VariableDeclaration.h"

namespace PTX {

class DeclarationStatement : public Statement
{
public:
	DeclarationStatement(VariableDeclaration *declaration) : m_declaration(declaration) {}

	// Properties

	const VariableDeclaration *GetDeclaration() const { return m_declaration; }
	VariableDeclaration *GetDeclaration() { return m_declaration; }
	void SetDeclaration(VariableDeclaration *declaration) { m_declaration = declaration; }

	// Formatting

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::Declaration";
		j["declaration"] = m_declaration->ToJSON();
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_declaration->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_declaration->Accept(visitor);
		}
		visitor.VisitOut(this);
	}
 
protected:
	VariableDeclaration *m_declaration = nullptr;
};

}
