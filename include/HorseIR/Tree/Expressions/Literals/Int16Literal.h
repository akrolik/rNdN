#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <cstdint>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class Int16Literal : public Literal<std::int16_t>
{
public:
	Int16Literal(std::int16_t value) : Literal<std::int16_t>(value, new BasicType(BasicType::Kind::Int16)) {}
	Int16Literal(const std::vector<std::int16_t>& values) : Literal<std::int16_t>(values, new BasicType(BasicType::Kind::Int16)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
