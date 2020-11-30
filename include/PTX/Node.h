#pragma once

#include <string>

#include "Libraries/json.hpp"

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

namespace PTX {

class Node
{
public:
	virtual std::string ToString(unsigned int indentation) const = 0;
	virtual json ToJSON() const = 0;

	virtual void Accept(ConstHierarchicalVisitor &visitor) const = 0;
};

}
