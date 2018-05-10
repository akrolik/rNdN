#pragma once

#include "PTX/Operands/Register/Register.h"
#include "PTX/Type.h"

namespace PTX {

class PredicateRegister : public Register<PredicateType>
{
public:
	using Register<PredicateType>::Register;
};

}
