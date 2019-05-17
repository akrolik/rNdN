#pragma once

#include <vector>

#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/ComplexValue.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class ComplexLiteral : public TypedVectorLiteral<ComplexValue *>
{
public:
	ComplexLiteral(ComplexValue *value) : TypedVectorLiteral<ComplexValue *>(value) {}
	ComplexLiteral(const std::vector<ComplexValue *>& values) : TypedVectorLiteral<ComplexValue *>(values) {}

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
