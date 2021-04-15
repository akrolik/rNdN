#pragma once

#include <string>

#include "SASS/Traversal/Visitor.h"

namespace SASS {

class Node
{
public:
	virtual std::string ToString() const = 0;

	// Visitors

	virtual void Accept(Visitor& visitor) = 0;
};

}
