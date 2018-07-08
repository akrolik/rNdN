#pragma once

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Types/PrimitiveType.h"

namespace HorseIR {

class Expression : public Node
{
public:
	virtual const Type *GetType() const = 0;
};

}
