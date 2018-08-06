#pragma once

#include <string>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Node
{
public:
	virtual std::string ToString() const = 0;

	virtual void Accept(Visitor &visitor) = 0;
	virtual void Accept(ConstVisitor &visitor) const = 0;
};

}
