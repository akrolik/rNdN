#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <cstdint>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class Int32Literal : public Literal<std::int32_t>
{
public:
	Int32Literal(std::int32_t value) : Literal<std::int32_t>(value, new BasicType(BasicType::Kind::Int32)) {}
	Int32Literal(const std::vector<std::int32_t>& values) : Literal<std::int32_t>(values, new BasicType(BasicType::Kind::Int32)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
