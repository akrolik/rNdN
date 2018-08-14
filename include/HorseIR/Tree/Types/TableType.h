#pragma once

#include <string>

#include "HorseIR/Tree/Types/BasicType.h"
#include "HorseIR/Tree/Types/DictionaryType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class TableType : public DictionaryType
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Table;

	TableType() : DictionaryType(TypeKind, new BasicType(BasicType::Kind::Symbol), new ListType(new BasicType(BasicType::Kind::Wildcard))) {}

	std::string ToString() const override
	{
		return "table";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const TableType& other) const
	{
		return true;
	}

	bool operator!=(const TableType& other) const
	{
		return false;
	}
};

}
