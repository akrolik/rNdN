#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR { 

class ModuleContent : public Node
{
public:
	virtual ModuleContent *Clone() const override = 0;
};

}
