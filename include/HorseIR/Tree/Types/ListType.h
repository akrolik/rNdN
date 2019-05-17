#pragma once

#include <vector>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class ListType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::List;

	ListType(Type *elementTypes) : ListType(TypeKind, {elementTypes}) {}
	ListType(const std::vector<Type *>& elementTypes) : ListType(TypeKind, elementTypes) {}

	const std::vector<Type *>& GetElementTypes() const { return m_elementTypes; }

	bool operator==(const ListType& other) const
	{
		return (m_elementTypes == other.m_elementTypes);
	}

	bool operator!=(const ListType& other) const
	{
		return (m_elementTypes != other.m_elementTypes);
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& elementType : m_elementTypes)
			{
				elementType->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& elementType : m_elementTypes)
			{
				elementType->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	ListType(Type::Kind kind, const std::vector<Type *>& elementTypes) : Type(kind), m_elementTypes(elementTypes) {}

	std::vector<Type *> m_elementTypes;
};

}
