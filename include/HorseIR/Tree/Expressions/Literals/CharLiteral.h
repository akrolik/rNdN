#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class CharLiteral : public TypedVectorLiteral<std::int8_t>
{
public:
	CharLiteral(std::int8_t value) : TypedVectorLiteral<std::int8_t>(value, BasicType::BasicKind::Char) {}
	CharLiteral(const std::vector<std::int8_t>& values) : TypedVectorLiteral<std::int8_t>(values, BasicType::BasicKind::Char) {}

	CharLiteral *Clone() const override
	{
		return new CharLiteral(m_values);
	}

	// Operators

	bool operator==(const CharLiteral& other) const
	{
		return (m_values == other.m_values);
	}

	bool operator!=(const CharLiteral& other) const
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
