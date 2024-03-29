#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class DictionaryType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Dictionary;

	DictionaryType(Type *keyType, Type *valueType) : DictionaryType(TypeKind, keyType, valueType) {}

	DictionaryType *Clone() const override
	{
		return new DictionaryType(m_keyType->Clone(), m_valueType->Clone());
	}

	// Key type

	const Type *GetKeyType() const { return m_keyType; }
	Type *GetKeyType() { return m_keyType; }

	void SetKeyType(Type *keyType) { m_keyType = keyType; }

	// Value type

	const Type *GetValueType() const { return m_valueType; }
	Type *GetValueType() { return m_valueType; }

	void SetValueType(Type *valueType) { m_valueType = valueType; }

	// Operators

	bool operator==(const DictionaryType& other) const
	{
		return (*m_keyType == *other.m_keyType && *m_valueType == *other.m_valueType);
	}

	bool operator!=(const DictionaryType& other) const
	{
		return (*m_keyType != *other.m_keyType || *m_valueType != *other.m_valueType);
	}

	// Visitors

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
	DictionaryType(Type::Kind kind, Type *keyType, Type *valueType) : Type(kind), m_keyType(keyType), m_valueType(valueType) {}

	Type *m_keyType = nullptr;
	Type *m_valueType = nullptr;
};

}
