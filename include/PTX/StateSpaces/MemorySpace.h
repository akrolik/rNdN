#pragma once

#include "PTX/StateSpaces/StateSpace.h"

namespace PTX {

template<class T, VectorSize V>
class Variable;

template<class T, VectorSize V = Scalar>
class MemorySpace : public StateSpace<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	using StateSpace<T, V>::StateSpace;

	Variable<T, V> *GetVariable(unsigned int index, unsigned int element = 0) const;
	Variable<T, V> *GetVariable(std::string name) const;

	virtual AddressSpace GetAddressSpace() const = 0;
private:
	using StateSpace<T, V>::m_elements;
};

template<class T, VectorSize V>
Variable<T, V> *MemorySpace<T, V>::GetVariable(unsigned int index, unsigned int element) const
{
	return new Variable<T, V>(m_elements.at(element), index);
}

template<class T, VectorSize V>
Variable<T, V> *MemorySpace<T, V>::GetVariable(std::string name) const
{
	for (typename std::vector<typename StateSpace<T, V>::Element *>::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it)
	{
		if ((*it)->GetName(0) == name)
		{
			return new Variable<T, V>(*it, 0);
		}
	}
	return nullptr;
}

}
