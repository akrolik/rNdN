#pragma once

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class EnumerationType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Enumeration;

	EnumerationType(Type *elementType) : Type(TypeKind), m_elementType(elementType) {}

	EnumerationType *Clone() const override
	{
		return new EnumerationType(m_elementType->Clone());
	}

	// Element type

	const Type *GetElementType() const { return m_elementType; }
	Type *GetElementType() { return m_elementType; }

	void SetElementType(Type *elementType) { m_elementType = elementType; }

	// Operators

	bool operator==(const EnumerationType& other) const
	{
		return (*m_elementType == *other.m_elementType);
	}

	bool operator!=(const EnumerationType& other) const
	{
		return (*m_elementType != *other.m_elementType);
	}

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_elementType->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_elementType->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	Type *m_elementType = nullptr;
};

}
