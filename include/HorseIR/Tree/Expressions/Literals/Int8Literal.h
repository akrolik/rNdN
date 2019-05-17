#pragma once

#include <cstdint>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class Int8Literal : public TypedVectorLiteral<std::int8_t>
{
public:
	Int8Literal(std::int8_t value) : TypedVectorLiteral<std::int8_t>(value) {}
	Int8Literal(const std::vector<std::int8_t>& values) : TypedVectorLiteral<std::int8_t>(values) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}
};

}
