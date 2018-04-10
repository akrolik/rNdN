#pragma once

#include <string>

#include "PTX/Operand.h"
#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class Register : public Operand<T, V>
{
public:
	Register(RegisterSpace<T, V> *space, unsigned int index) : m_space(space), m_index(index) {}

	std::string ToString()
	{
		return "%" + m_space->Name(m_index);
	}

	std::string Name()
	{
		return m_space->Name(m_index);
	}

protected:
	RegisterSpace<T, V> *m_space = nullptr;
	unsigned int m_index = -1;
};

}
