#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class BooleanLiteral : public TypedVectorLiteral<bool>
{
public:
	BooleanLiteral(bool value) : TypedVectorLiteral<bool>(value, BasicType::BasicKind::Boolean) {}
	BooleanLiteral(const std::vector<bool>& values) : TypedVectorLiteral<bool>(values, BasicType::BasicKind::Boolean) {}

	BooleanLiteral *Clone() const override
	{
		return new BooleanLiteral(m_values);
	}

	bool operator==(const BooleanLiteral& other) const
	{
		return (m_values == other.m_values);
	}

	bool operator!=(const BooleanLiteral& other) const
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

static VectorLiteral *CreateBooleanLiteral(const std::vector<std::int64_t>& values)
{
	std::vector<bool> converted;
	for (std::int64_t value : values)
	{
		converted.push_back(static_cast<bool>(value));
	}
	return new BooleanLiteral(converted);
}

static VectorLiteral *CreateBooleanLiteral(std::int64_t value)
{
	std::vector<std::int64_t> values = { value };
	return CreateBooleanLiteral(values);
}

}
