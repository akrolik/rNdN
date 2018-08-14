#pragma once

#include <string>

#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/TableType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class KeyedTableType : public ListType
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::KeyedTable;

	KeyedTableType() : ListType(TypeKind, new TableType()) {}

	std::string ToString() const override
	{
		return "ktable";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const KeyedTableType& other) const
	{
		return true;
	}

	bool operator!=(const KeyedTableType& other) const
	{
		return false;
	}
};

}
