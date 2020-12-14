#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class WildcardType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Wildcard;

	WildcardType() : Type(Type::Kind::Wildcard) {}

	WildcardType *Clone() const override
	{
		return new WildcardType();
	}

	// Operators

	bool operator==(const WildcardType& other) const
	{
		return false;
	}

	bool operator!=(const WildcardType& other) const
	{
		return true;
	}

	// Visitors

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}
};

}
