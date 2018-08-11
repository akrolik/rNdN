#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class DictionaryType : public Type
{
public:
	DictionaryType(Type *keyType, Type *valueType) : Type(Type::Kind::Dictionary), m_keyType(keyType), m_valueType(valueType) {}

	Type *GetKeyType() const { return m_keyType; }
	Type *GetValueType() const { return m_valueType; }

	std::string ToString() const override
	{
		return "dict<" + m_keyType->ToString() + ", " + m_valueType->ToString() + ">";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const DictionaryType& other) const
	{
		return (*m_keyType == *other.m_keyType && *m_valueType == *other.m_valueType);
	}

	bool operator!=(const DictionaryType& other) const
	{
		return (*m_keyType != *other.m_keyType || *m_valueType != *other.m_valueType);
	}

private:
	Type *m_keyType = nullptr;
	Type *m_valueType = nullptr;
};

}
