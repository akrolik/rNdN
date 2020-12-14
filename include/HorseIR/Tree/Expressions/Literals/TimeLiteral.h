#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/TimeValue.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class TimeLiteral : public TypedVectorLiteral<TimeValue *>
{
public:
	TimeLiteral(TimeValue *value) : TypedVectorLiteral<TimeValue *>(value, BasicType::BasicKind::Time) {}
	TimeLiteral(const std::vector<TimeValue *>& values) : TypedVectorLiteral<TimeValue *>(values, BasicType::BasicKind::Time) {}

	TimeLiteral *Clone() const override
	{
		std::vector<TimeValue *> values;
		for (const auto& value : m_values)
		{
			values.push_back(value->Clone());
		}
		return new TimeLiteral(values);
	}

	// Operators

	bool operator==(const TimeLiteral& other) const
	{
		return std::equal(
			std::begin(m_values), std::end(m_values),
			std::begin(other.m_values), std::end(other.m_values),
			[](const TimeValue *v1, const TimeValue *v2) { return *v1 == *v2; }
		);
	}

	bool operator!=(const TimeLiteral& other) const
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
