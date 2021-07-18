#pragma once

#include "SASS/Tree/Operands/Operand.h"

namespace SASS {

class Composite : public Operand
{
public:
	using Operand::Operand;

	virtual bool GetOpModifierNegate() const { return false; }
};

}
