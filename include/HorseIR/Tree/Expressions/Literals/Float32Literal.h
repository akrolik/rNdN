#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class Float32Literal : public TypedVectorLiteral<float>
{
public:
	Float32Literal(float value) : TypedVectorLiteral<float>(value, BasicType::BasicKind::Float32) {}
	Float32Literal(const std::vector<float>& values) : TypedVectorLiteral<float>(values, BasicType::BasicKind::Float32) {}

	Float32Literal *Clone() const override
	{
		return new Float32Literal(m_values);
	}

	bool operator==(const Float32Literal& other) const
	{
		return (m_values == other.m_values);
	}

	bool operator!=(const Float32Literal& other) const
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
