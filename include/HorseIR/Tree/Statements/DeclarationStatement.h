#pragma once

#include <string>
#include <vector>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Tree/VariableDeclaration.h"
#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class DeclarationStatement : public Statement
{
public:
	DeclarationStatement(VariableDeclaration *declaration) : m_declaration(declaration) {}

	DeclarationStatement *Clone() const override
	{
		return new DeclarationStatement(m_declaration->Clone());
	}

	// Declaration

	const VariableDeclaration *GetDeclaration() const { return m_declaration; }
	VariableDeclaration *GetDeclaration() { return m_declaration; }

	void SetDeclaration(VariableDeclaration *declaration) { m_declaration = declaration; }

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_declaration->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
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

static std::vector<Statement *> *CreateDeclarationStatements(const std::vector<std::string>& names, Type *type)
{
	auto statements = new std::vector<Statement *>();
	for (auto& name : names)
	{
		auto statement = new DeclarationStatement(new VariableDeclaration(name, type));
		statements->push_back(statement);
	}
	return statements;
}

}
