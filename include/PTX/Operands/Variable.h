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
	Variable(typename MemorySpace<T, V>::Element *element, unsigned int index = 0) : m_structure(element), m_index(index) {}

	virtual AddressSpace GetAddressSpace() const
	{
		//TODO: Return the actual addrss space
		// return m_structure->GetAddressSpace();
		return AddressSpace::Param;
	}

	virtual std::string GetName() const
	{
		return m_structure->GetName(m_index);
	}

	std::string ToString() const
	{
		return GetName();
	}

protected:
	// typename MemorySpace<T, V>::Element *m_structure = nullptr;
	Structure *m_structure = nullptr;
	unsigned int m_index;
};

}
