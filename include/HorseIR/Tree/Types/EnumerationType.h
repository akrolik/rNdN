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

	EnumerationType(Type *type) : Type(TypeKind), m_keyType(type), m_valueType(type) {}
	EnumerationType(Type *keyType, Type *valueType) : Type(TypeKind), m_keyType(keyType), m_valueType(valueType) {}

	Type *GetKeyType() const { return m_keyType; }
	Type *GetValueType() const { return m_valueType; }

	bool operator==(const EnumerationType& other) const
	{
		return (*m_keyType == *other.m_keyType && *m_valueType == *other.m_valueType);
	}

	bool operator!=(const EnumerationType& other) const
	{
		return (*m_keyType != *other.m_keyType || *m_valueType != *other.m_valueType);
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			m_keyType->Accept(visitor);
			m_valueType->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			m_keyType->Accept(visitor);
			m_valueType->Accept(visitor);
		}
		visitor.VisitOut(this);
	}

protected:
	Type *m_keyType = nullptr;
	Type *m_valueType = nullptr;
};

}
