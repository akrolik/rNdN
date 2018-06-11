#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class WildcardType : public Type
{
public:
	std::string ToString() const override { return "?"; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
};

}
