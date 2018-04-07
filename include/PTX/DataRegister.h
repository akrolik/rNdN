#pragma once

#include "PTX/Register.h"
#include "PTX/Type.h"

namespace PTX {

template<Type T>
class DataRegister : Register
{
public:
	std::string m_name;
	unsigned int m_index = -1;

private:
};

}
