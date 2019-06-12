#pragma once

#include "HorseIR/Tree/Types/WildcardType.h"
#include "HorseIR/Tree/Types/BasicType.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/DictionaryType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class TableType : public DictionaryType
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Table;

	TableType() : DictionaryType(TypeKind, new BasicType(BasicType::BasicKind::Symbol), new ListType(new WildcardType())) {}

	bool operator==(const TableType& other) const
	{
		return true;
	}

	bool operator!=(const TableType& other) const
	{
		return false;
	}

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
