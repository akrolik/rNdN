#pragma once

#include <string>

#include "PTX/Operands/Operand.h"
#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class Register : public Operand<T, V>
{
public:
	Register(typename RegisterSpace<T, V>::Element *element, unsigned int index = 0) : m_element(element), m_index(index) {}

	virtual std::string Name()
	{
		return m_element->VariableName(m_index);
	}

	std::string ToString()
	{
		return m_element->Name(m_index);
	}

protected:
	typename RegisterSpace<T, V>::Element *m_element = nullptr;
	unsigned int m_index;
};

}
