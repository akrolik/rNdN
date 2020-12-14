#pragma once

#include <cstdint>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/DatetimeValue.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class DatetimeLiteral : public TypedVectorLiteral<DatetimeValue *>
{
public:
	DatetimeLiteral(DatetimeValue *value) : TypedVectorLiteral<DatetimeValue *>(value, BasicType::BasicKind::Datetime) {}
	DatetimeLiteral(const std::vector<DatetimeValue *>& values) : TypedVectorLiteral<DatetimeValue *>(values, BasicType::BasicKind::Datetime) {}

	DatetimeLiteral *Clone() const override
	{
		std::vector<DatetimeValue *> values;
		for (const auto& value : m_values)
		{
			values.push_back(value->Clone());
		}
		return new DatetimeLiteral(values);
	}

	// Operators

	bool operator==(const DatetimeLiteral& other) const
	{
		return std::equal(
			std::begin(m_values), std::end(m_values),
			std::begin(other.m_values), std::end(other.m_values),
			[](const DatetimeValue *v1, const DatetimeValue *v2) { return *v1 == *v2; }
		);
	}

	bool operator!=(const DatetimeLiteral& other) const
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
