#pragma once

#include <cstdint>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/MonthValue.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class MonthLiteral : public TypedVectorLiteral<MonthValue *>
{
public:
	MonthLiteral(MonthValue *value) : TypedVectorLiteral<MonthValue *>(value, BasicType::BasicKind::Month) {}
	MonthLiteral(const std::vector<MonthValue *>& values) : TypedVectorLiteral<MonthValue *>(values, BasicType::BasicKind::Month) {}

	MonthLiteral *Clone() const override
	{
		std::vector<MonthValue *> values;
		for (const auto& value : m_values)
		{
			values.push_back(value->Clone());
		}
		return new MonthLiteral(values);
	}

	// Operators

	bool operator==(const MonthLiteral& other) const
	{
		return std::equal(
			std::begin(m_values), std::end(m_values),
			std::begin(other.m_values), std::end(other.m_values),
			[](const MonthValue *v1, const MonthValue *v2) { return *v1 == *v2; }
		);
	}

	bool operator!=(const MonthLiteral& other) const
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
