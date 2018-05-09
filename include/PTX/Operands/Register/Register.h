#pragma once

#include <string>

#include "PTX/Operands/Operand.h"
#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class Register : public Operand<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	Register(typename RegisterSpace<T, V>::Element *element, unsigned int index = 0) : m_element(element), m_index(index) {}

	virtual std::string GetName() const
	{
		return m_element->GetName(m_index);
	}

	std::string ToString() const
	{
		return GetName();
	}

protected:
	typename RegisterSpace<T, V>::Element *m_element = nullptr;
	unsigned int m_index;
};

}
