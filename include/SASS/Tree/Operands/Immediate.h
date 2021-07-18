#pragma once

#include "SASS/Tree/Operands/Composite.h"

namespace SASS {

class Immediate : public Composite
{
public:
	Immediate() : Composite(Operand::Kind::Immediate) {}
};

template<unsigned int N>
class SizedImmediate : public Immediate
{

};

}
