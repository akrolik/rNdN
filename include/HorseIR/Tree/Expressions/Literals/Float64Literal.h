#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class Float64Literal : public Literal<double>
{
public:
	Float64Literal(double value) : Literal<double>(value, new BasicType(BasicType::Kind::Float64)) {}
	Float64Literal(const std::vector<double>& values) : Literal<double>(values, new BasicType(BasicType::Kind::Float64)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
