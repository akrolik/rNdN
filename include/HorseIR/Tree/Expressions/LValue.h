#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR {

class LValue : virtual public Node
{
protected:
	LValue() {}
};

}
