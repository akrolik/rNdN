#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <cstdint>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class Int64Literal : public Literal<std::int64_t>
{
public:
	Int64Literal(std::int64_t value) : Literal<std::int64_t>(value, new BasicType(BasicType::Kind::Int64)) {}
	Int64Literal(const std::vector<std::int64_t>& values) : Literal<std::int64_t>(values, new BasicType(BasicType::Kind::Int64)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
