#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "PTX/StateSpaces/StateSpace.h"

namespace PTX {

template<class T, VectorSize V>
class Register;

template<class T, VectorSize V>
class IndexedRegister;

template<class T, VectorSize V = Scalar>
class RegisterSpace : public StateSpace<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	RegisterSpace(std::string prefix, unsigned int count) : StateSpace<T, V>(prefix, count) {}
	RegisterSpace(std::string name) : StateSpace<T, V>(name) {}
	RegisterSpace(std::vector<std::string> names) : StateSpace<T, V>(names) {}

	Register<T, V> *GetRegister(std::string name) const;
	Register<T, V> *GetRegister(unsigned int index, unsigned int element = 0) const;

	IndexedRegister<T, V> *GetRegister(unsigned int index, VectorElement vectorElement, unsigned int element = 0) const;
	IndexedRegister<T, V> *GetRegister(std::string name, VectorElement vectorElement, unsigned int element = 0) const;

	std::string Specifier() const { return ".reg"; }
private:
	using StateSpace<T, V>::m_elements;
};

#include "PTX/Operands/Register/Register.h"
// #include "PTX/Operands/IndexedRegister.h"

template<class T, VectorSize V>
Register<T, V> *RegisterSpace<T, V>::GetRegister(unsigned int index, unsigned int element) const
{
	return new Register<T, V>(m_elements.at(element), index);
}

template<class T, VectorSize V>
Register<T, V> *RegisterSpace<T, V>::GetRegister(std::string name) const
{
	for (typename std::vector<typename StateSpace<T, V>::Element *>::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it)
	{
		if ((*it)->GetName(0) == name)
		{
			return new Register<T, V>(*it, 0);
		}
	}
	return nullptr;
}

template<class T, VectorSize V>
IndexedRegister<T, V> *RegisterSpace<T, V>::GetRegister(unsigned int index, VectorElement vectorElement, unsigned int element) const
{
	return new IndexedRegister<T, V>(m_elements.at(element), index, vectorElement);
}

template<class T, VectorSize V>
IndexedRegister<T, V> *RegisterSpace<T, V>::GetRegister(std::string name, VectorElement vectorElement, unsigned int element) const
{
	for (typename std::vector<typename StateSpace<T, V>::Element *>::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it)
	{
		if ((*it)->GetName(0) == name)
		{
			return new IndexedRegister<T, V>(*it, 0, vectorElement);
		}
	}
	return nullptr;
}

}
