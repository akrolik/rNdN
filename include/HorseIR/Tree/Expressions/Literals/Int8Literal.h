#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <cstdint>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class Int8Literal : public Literal<std::int8_t>
{
public:
	Int8Literal(std::int8_t value) : Literal<std::int8_t>(value, new BasicType(BasicType::Kind::Int8)) {}
	Int8Literal(const std::vector<std::int8_t>& values) : Literal<std::int8_t>(values, new BasicType(BasicType::Kind::Int8)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
