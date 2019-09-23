#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR {

class Statement : public Node
{
public:
	virtual Statement *Clone() const override = 0;
};

}
