#pragma once

#include "PTX/Register.h"

namespace PTX {

class PredicateRegister : Register
{
public:
	std::string m_name;
	unsigned int m_index = -1;

private:
};

}
