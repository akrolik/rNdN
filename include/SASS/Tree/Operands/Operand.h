#pragma once

#include "SASS/Tree/Node.h"

namespace SASS {

class Operand : public Node
{
public:
	// Binary/Formatting

	virtual std::string ToString() const = 0;
	virtual std::uint64_t ToBinary() const = 0;
};

}
