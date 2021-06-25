#pragma once

#include "SASS/Tree/Operands/Composite.h"

namespace SASS {

class Immediate : public Composite
{
public:
	OpCodeKind GetOpCodeKind() const override { return OpCodeKind::Immediate; }
};

template<unsigned int N>
class SizedImmediate : public Immediate
{

};

}
