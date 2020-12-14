#pragma once

#include <string>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Tree/VariableDeclaration.h"
#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class GlobalDeclaration : public ModuleContent
{
public:
	GlobalDeclaration(VariableDeclaration *declaration, Operand *expression) : m_declaration(declaration), m_expression(expression) {}

	GlobalDeclaration *Clone() const override
	{
		return new GlobalDeclaration(m_declaration->Clone(), m_expression->Clone());
	}

	// Properties

	const VariableDeclaration *GetDeclaration() const { return m_declaration; }
	VariableDeclaration *GetDeclaration() { return m_declaration; }
	void SetDeclaration(VariableDeclaration *declaration) { m_declaration = declaration; }

	const Operand *GetExpression() const { return m_expression; }
	Operand *GetExpression() { return m_expression; }
	void SetExpression(Operand *expression) { m_expression = expression; }

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_declaration->Accept(visitor);
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_declaration->Accept(visitor);
			m_expression->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	VariableDeclaration *m_declaration = nullptr;
	Operand *m_expression = nullptr;
};

}
