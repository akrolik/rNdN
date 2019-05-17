#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class BooleanLiteral : public TypedVectorLiteral<bool>
{
public:
	BooleanLiteral(bool value) : TypedVectorLiteral<bool>(value) {}
	BooleanLiteral(const std::vector<bool>& values) : TypedVectorLiteral<bool>(values) {}

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
