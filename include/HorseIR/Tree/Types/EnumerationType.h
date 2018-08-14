#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class EnumerationType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Enumeration;

	EnumerationType(Type *valueType, Type *keyType) : Type(TypeKind), m_valueType(valueType), m_keyType(keyType) {}

	Type *GetValueType() const { return m_valueType; }
	Type *GetKeyType() const { return m_keyType; }

	std::string ToString() const override
	{
		return "enum<" + m_valueType->ToString() + ", " + m_keyType->ToString() + ">";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const EnumerationType& other) const
	{
		return (*m_valueType == *other.m_valueType && *m_keyType == *other.m_keyType);
	}

	bool operator!=(const EnumerationType& other) const
	{
		return (*m_valueType != *other.m_valueType || *m_keyType != *other.m_keyType);
	}

protected:
	Type *m_valueType = nullptr;
	Type *m_keyType = nullptr;
};

}
