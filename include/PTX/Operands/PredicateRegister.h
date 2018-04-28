#pragma once

#include "PTX/Operands/Register.h"
#include "PTX/Type.h"

namespace PTX {

class PredicateRegister : public Register<PredicateType>
{
public:
	PredicateRegister(typename RegisterSpace<PredicateType>::Element *element, unsigned int index = 0) : Register(element, index) {}

	std::string VariableName()
	{
		return m_element->VariableName(0);
	}
};

}
