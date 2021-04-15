#pragma once

#include "SASS/Tree/Node.h"

namespace SASS {

class Operand : public Node
{
public:
	// Binary

	virtual std::uint64_t ToBinary() const = 0;
};

}
