#pragma once

#include <string>

#include "HorseIR/Tree/Types/TableType.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class TableType : public Type
{
public:
	std::string ToString() const override { return "table"; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
};

}
