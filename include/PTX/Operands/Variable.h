#pragma once

#include <string>

#include "PTX/Operands/Operand.h"
#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class Variable : public Operand<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	Variable(typename MemorySpace<T, V>::Element *element, unsigned int index = 0) : m_element(element), m_index(index) {}

	virtual AddressSpace GetAddressSpace() const
	{
		//TODO:
		// return m_element->GetAddressSpace();
		return AddressSpace::Param;
	}

	virtual std::string GetName() const
	{
		return m_element->GetName(m_index);
	}

	std::string ToString() const
	{
		return GetName();
	}

protected:
	typename MemorySpace<T, V>::Element *m_element = nullptr;
	unsigned int m_index;
};

}
