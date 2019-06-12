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

	EnumerationType(Type *type) : Type(TypeKind), m_elementType(type) {}

	Type *GetElementType() const { return m_elementType; }

	bool operator==(const EnumerationType& other) const
	{
		return (*m_elementType == *other.m_elementType);
	}

	bool operator!=(const EnumerationType& other) const
	{
		return (*m_elementType != *other.m_elementType);
	}

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
