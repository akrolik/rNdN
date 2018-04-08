#pragma once

#include "PTX/Register.h"

namespace PTX {

template<Type T>
class DataRegister : Register<T>
{
public:
	DataRegister(std::string name) : m_name(name) {}
	DataRegister(std::string prefixm, unsigned int index) : m_name(prefix), m_index(index) {}

private:
	std::string m_name;
	unsigned int m_index = -1;
};

}
