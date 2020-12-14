#pragma once

#include <cstdint>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/DateValue.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class DateLiteral : public TypedVectorLiteral<DateValue *>
{
public:
	DateLiteral(DateValue *value) : TypedVectorLiteral<DateValue *>(value, BasicType::BasicKind::Date) {}
	DateLiteral(const std::vector<DateValue *>& values) : TypedVectorLiteral<DateValue *>(values, BasicType::BasicKind::Date) {}

	DateLiteral *Clone() const override
	{
		std::vector<DateValue *> values;
		for (const auto& value : m_values)
		{
			values.push_back(value->Clone());
		}
		return new DateLiteral(values);
	}

	// Operators

	bool operator==(const DateLiteral& other) const
	{
		return std::equal(
			std::begin(m_values), std::end(m_values),
			std::begin(other.m_values), std::end(other.m_values),
			[](const DateValue *v1, const DateValue *v2) { return *v1 == *v2; }
		);
	}

	bool operator!=(const DateLiteral& other) const
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
