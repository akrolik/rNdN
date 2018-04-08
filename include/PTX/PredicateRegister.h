#pragma once

#include "PTX/Register.h"
#include "PTX/Type.h"

namespace PTX {

class PredicateRegister : Register<Predicate>
{
public:
	PredicateRegister(std::string name) : Register(name) {}
	PredicateRegister(std::string prefix, unsigned int index) : Register(prefix, index) {}

private:
};

}
