#pragma once

#include "HorseIR/Tree/Expressions/Operand.h"

namespace HorseIR {

class Literal : public Operand
{
protected:
	Literal() : Operand(Operand::Kind::Literal) {}
};

}
