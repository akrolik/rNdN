#pragma once

#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/TableType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class KeyedTableType : public ListType
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::KeyedTable;

	KeyedTableType() : ListType(TypeKind, {new TableType(), new TableType()}) {}

	bool operator==(const KeyedTableType& other) const
	{
		return true;
	}

	bool operator!=(const KeyedTableType& other) const
	{
		return false;
	}

	Type *GetKeyType() const { return m_elementTypes.at(0); }
	Type *GetValueType() const { return m_elementTypes.at(1); }

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
