#pragma once

#include <cstdint>
#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/MonthValue.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class MonthLiteral : public TypedVectorLiteral<MonthValue *>
{
public:
	MonthLiteral(MonthValue *value) : TypedVectorLiteral<MonthValue *>(value) {}
	MonthLiteral(const std::vector<MonthValue *>& values) : TypedVectorLiteral<MonthValue *>(values) {}

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
