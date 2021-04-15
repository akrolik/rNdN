#pragma once

#include "SASS/Tree/Operands/Operand.h"

namespace SASS {

class Immediate : public Composite
{
public:
	OpCodeKind GetOpCodeKind() const override { return OpCodeKind::Immediate; }
};

}
