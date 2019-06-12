#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class Float64Literal : public TypedVectorLiteral<double>
{
public:
	Float64Literal(double value) : TypedVectorLiteral<double>(value, BasicType::BasicKind::Float64) {}
	Float64Literal(const std::vector<double>& values) : TypedVectorLiteral<double>(values, BasicType::BasicKind::Float64) {}

	bool operator==(const Float64Literal& other) const
	{
		return (m_values == other.m_values);
	}

	bool operator!=(const Float64Literal& other) const
	{
		return !(*this == other);
	}

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
