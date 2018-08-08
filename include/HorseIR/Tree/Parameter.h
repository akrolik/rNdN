#pragma once

#include <string>

#include "HorseIR/Tree/Declaration.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Parameter : public Declaration
{
public:
	using Declaration::Declaration;

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
