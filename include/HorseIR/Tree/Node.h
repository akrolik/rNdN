#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class Node
{
public:
	virtual void Accept(Visitor &visitor) = 0;
	virtual void Accept(ConstVisitor &visitor) const = 0;

	virtual void Accept(HierarchicalVisitor &visitor) = 0;
	virtual void Accept(ConstHierarchicalVisitor &visitor) const = 0;
};

}
