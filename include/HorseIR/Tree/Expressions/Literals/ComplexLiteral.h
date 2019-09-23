#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/ComplexValue.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class ComplexLiteral : public TypedVectorLiteral<ComplexValue *>
{
public:
	ComplexLiteral(ComplexValue *value) : TypedVectorLiteral<ComplexValue *>(value, BasicType::BasicKind::Complex) {}
	ComplexLiteral(const std::vector<ComplexValue *>& values) : TypedVectorLiteral<ComplexValue *>(values, BasicType::BasicKind::Complex) {}

	ComplexLiteral *Clone() const override
	{
		std::vector<ComplexValue *> values;
		for (const auto& value : m_values)
		{
			values.push_back(value->Clone());
		}
		return new ComplexLiteral(values);
	}

	bool operator==(const ComplexLiteral& other) const
	{
		return std::equal(
			std::begin(m_values), std::end(m_values),
			std::begin(other.m_values), std::end(other.m_values),
			[](const ComplexValue *v1, const ComplexValue *v2) { return *v1 == *v2; }
		);
	}

	bool operator!=(const ComplexLiteral& other) const
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
