#pragma once

#include <string>

#include "PTX/Operand.h"

namespace PTX {

template<typename T>
class Register : Operand<T>
{
public:
	Register(std::string name) : m_name(name) {}
	Register(std::string prefix, unsigned int index) : m_name(prefix), m_index(index) {}

	inline std::string ToString()
	{
		if (m_index >= 0)
		{
			return "%" + m_name + std::to_string(m_index);
		}
		return "%" + m_name;
	}

private:
	std::string m_name;
	unsigned int m_index = -1;
};

}
