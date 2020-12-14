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

	KeyedTableType *Clone() const override
	{
		return new KeyedTableType();
	}

	// Key type

	const Type *GetKeyType() const { return m_elementTypes.at(0); }
	Type *GetKeyType() { return m_elementTypes.at(0); }

	void SetKeyType(Type *type) { m_elementTypes.at(0) = type; }

	// Value type

	const Type *GetValueType() const { return m_elementTypes.at(1); }
	Type *GetValueType() { return m_elementTypes.at(1); }

	void SetValueType(Type *type) { m_elementTypes.at(1) = type; }

	// Operators

	bool operator==(const KeyedTableType& other) const
	{
		return true;
	}

	bool operator!=(const KeyedTableType& other) const
	{
		return false;
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
