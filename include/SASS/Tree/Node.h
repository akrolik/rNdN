#pragma once

#include <string>

#include "SASS/Traversal/Visitor.h"
#include "SASS/Traversal/ConstVisitor.h"

namespace SASS {

class Node
{
public:
	// Visitors

	virtual void Accept(Visitor& visitor) = 0;
	virtual void Accept(ConstVisitor& visitor) const = 0;
};

}
