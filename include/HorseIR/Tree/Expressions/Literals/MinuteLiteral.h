#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/MinuteValue.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class MinuteLiteral : public TypedVectorLiteral<MinuteValue *>
{
public:
	MinuteLiteral(MinuteValue *value) : TypedVectorLiteral<MinuteValue *>(value, BasicType::BasicKind::Minute) {}
	MinuteLiteral(const std::vector<MinuteValue *>& values) : TypedVectorLiteral<MinuteValue *>(values, BasicType::BasicKind::Minute) {}

	MinuteLiteral *Clone() const override
	{
		std::vector<MinuteValue *> values;
		for (const auto& value : m_values)
		{
			values.push_back(value->Clone());
		}
		return new MinuteLiteral(values);
	}

	// Operators

	bool operator==(const MinuteLiteral& other) const
	{
		return std::equal(
			std::begin(m_values), std::end(m_values),
			std::begin(other.m_values), std::end(other.m_values),
			[](const MinuteValue *v1, const MinuteValue *v2) { return *v1 == *v2; }
		);
	}

	bool operator!=(const MinuteLiteral& other) const
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
