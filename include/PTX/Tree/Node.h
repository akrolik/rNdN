#pragma once

#include "Libraries/json.hpp"

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstVisitor.h"
#include "PTX/Traversal/HierarchicalVisitor.h"
#include "PTX/Traversal/Visitor.h"

namespace PTX {

class Node
{
public:
	virtual json ToJSON() const = 0;

	virtual void Accept(Visitor &visitor) = 0;
	virtual void Accept(ConstVisitor &visitor) const = 0;

	virtual void Accept(HierarchicalVisitor &visitor) = 0;
	virtual void Accept(ConstHierarchicalVisitor &visitor) const = 0;
};

}
