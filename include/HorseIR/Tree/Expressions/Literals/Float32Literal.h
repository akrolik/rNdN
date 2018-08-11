#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class Float32Literal : public Literal<float>
{
public:
	Float32Literal(float value) : Literal<float>(value, new BasicType(BasicType::Kind::Float32)) {}
	Float32Literal(const std::vector<float>& values) : Literal<float>(values, new BasicType(BasicType::Kind::Float32)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
