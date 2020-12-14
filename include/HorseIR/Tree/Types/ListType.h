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

	ListType *Clone() const override
	{
		std::vector<Type *> elementTypes;
		for (const auto& elementType : m_elementTypes)
		{
			elementTypes.push_back(elementType->Clone());
		}
		return new ListType(elementTypes);
	}

	// Element types

	std::vector<const Type *> GetElementTypes() const
	{
		return { std::begin(m_elementTypes), std::end(m_elementTypes) };
	}
	std::vector<Type *>& GetElementTypes() { return m_elementTypes; }

	void AddElementType(Type *elementType) { m_elementTypes.push_back(elementType); }
	void SetElementTypes(const std::vector<Type *>& elementTypes) { m_elementTypes = elementTypes; }

	// Operators

	bool operator==(const ListType& other) const
	{
		return std::equal(
			std::begin(m_elementTypes), std::end(m_elementTypes),
			std::begin(other.m_elementTypes), std::end(other.m_elementTypes),
			[](const Type *t1, const Type *t2) { return *t1 == *t2; }
		);
	}

	bool operator!=(const ListType& other) const
	{
		return !(*this == other);
	}

	// Visitors

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
