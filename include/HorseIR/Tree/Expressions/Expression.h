#pragma once

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Expression : public Node
{
public:
	virtual const Type *GetType() const = 0;
};

}
