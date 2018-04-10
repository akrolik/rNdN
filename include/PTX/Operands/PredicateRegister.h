#pragma once

#include "PTX/Operands/Register.h"
#include "PTX/Type.h"

namespace PTX {

class PredicateRegister : public Register<Predicate>
{
public:
	PredicateRegister(RegisterSpace<Predicate> *space, unsigned int index) : Register(space, index) {}
};

}
