#pragma once

#include <cstdint>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class Int32Literal : public TypedVectorLiteral<std::int32_t>
{
public:
	Int32Literal(std::int32_t value) : TypedVectorLiteral<std::int32_t>(value, BasicType::BasicKind::Int32) {}
	Int32Literal(const std::vector<std::int32_t>& values) : TypedVectorLiteral<std::int32_t>(values, BasicType::BasicKind::Int32) {}

	Int32Literal *Clone() const override
	{
		return new Int32Literal(m_values);
	}

	// Operators

	bool operator==(const Int32Literal& other) const
	{
		return (m_values == other.m_values);
	}

	bool operator!=(const Int32Literal& other) const
	{
		return !(*this == other);
	}

	// Visitors

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
