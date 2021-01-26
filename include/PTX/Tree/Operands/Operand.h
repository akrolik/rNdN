#pragma once

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Node.h"

#include "PTX/Traversal/Dispatch.h"
#include "PTX/Traversal/OperandVisitor.h"
#include "PTX/Traversal/ConstOperandVisitor.h"

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

	virtual void Accept(OperandVisitor& visitor) = 0;
	virtual void Accept(ConstOperandVisitor& visitor) const = 0;
};

template<class T, bool Assert = true>
class TypedOperand : public virtual Operand
{
public:
	REQUIRE_TYPE_PARAM(Operand,
		REQUIRE_BASE(T, DataType)
	);
};

}
