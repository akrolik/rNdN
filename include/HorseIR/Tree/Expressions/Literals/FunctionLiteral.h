#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include "HorseIR/Tree/FunctionDeclaration.h"
#include "HorseIR/Tree/Expressions/Identifier.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class FunctionLiteral : public Literal
{
public:
	FunctionLiteral(Identifier *identifier) : m_identifier(identifier) {}

	Identifier *GetIdentifier() const { return m_identifier; }
	void SetIdentifier(Identifier *identifier) { m_identifier = identifier; }

	FunctionDeclaration *GetFunction() const { return m_function; }
	void SetFunction(FunctionDeclaration *function) { m_function = function; }

	bool operator==(const FunctionLiteral& other) const
	{
		return (*m_identifier == *other.m_identifier);
	}

	bool operator!=(const FunctionLiteral& other) const
	{
		return !(*this == other);
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_identifier->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_identifier->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	Identifier *m_identifier = nullptr;

	FunctionDeclaration *m_function = nullptr;
};

}
