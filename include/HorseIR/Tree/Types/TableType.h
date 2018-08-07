#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class TableType : public Type
{
public:
	TableType() : Type(Type::Kind::Table) {}

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
