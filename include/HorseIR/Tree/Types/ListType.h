#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class ListType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::List;

	ListType(Type *elementType) : ListType(TypeKind, elementType) {}

	Type *GetElementType() const { return m_elementType; }

	std::string ToString() const override
	{
		return "list<" + m_elementType->ToString() + ">";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const ListType& other) const
	{
		return (*m_elementType == *other.m_elementType);
	}

	bool operator!=(const ListType& other) const
	{
		return (*m_elementType != *other.m_elementType);
	}

protected:
	ListType(Type::Kind kind, Type *elementType) : Type(kind), m_elementType(elementType) {}

	Type *m_elementType = nullptr;
};

}
