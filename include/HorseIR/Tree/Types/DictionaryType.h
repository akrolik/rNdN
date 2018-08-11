#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class DictionaryType : public Type
{
public:
	DictionaryType(const Type *keyType, const Type *valueType) : Type(Type::Kind::Dictionary), m_keyType(keyType), m_valueType(valueType) {}

	const Type *GetKeyType() const { return m_keyType; }
	const Type *GetValueType() const { return m_valueType; }

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
	const Type *m_keyType = nullptr;
	const Type *m_valueType = nullptr;
};

}
