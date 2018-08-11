#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class BoolLiteral : public Literal<bool>
{
public:
	BoolLiteral(bool value) : Literal<bool>(value, new BasicType(BasicType::Kind::Bool)) {}
	BoolLiteral(const std::vector<bool>& values) : Literal<bool>(values, new BasicType(BasicType::Kind::Bool)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
