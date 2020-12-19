#pragma once

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Node.h"

#include "PTX/Traversal/OperandVisitor.h"

namespace PTX {

class Operand : public Node
{
public:
	virtual std::string ToString() const = 0;

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	//TODO: Remove default
	virtual void Accept(OperandVisitor& visitor) const {}
};

template<class T>
class TypedOperand : public virtual Operand
{
public:
	REQUIRE_TYPE_PARAM(Operand,
		REQUIRE_BASE(T, Type)
	);
};

}
