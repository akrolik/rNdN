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
	using StateSpace<T, V>::StateSpace;

	Register<T, V> *GetRegister(std::string name, unsigned int index = 0) const;
	IndexedRegister<T, V> *GetRegister(std::string name, unsigned int index, VectorElement vectorElement) const;

	std::string Specifier() const { return ".reg"; }
private:
	using StateSpace<T, V>::GetStructure;
};

#include "PTX/Operands/Register/Register.h"

template<class T, VectorSize V>
Register<T, V> *RegisterSpace<T, V>::GetRegister(std::string name, unsigned int index) const
{
	return new Register<T, V>(GetStructure(name), index);
}

template<class T, VectorSize V>
IndexedRegister<T, V> *RegisterSpace<T, V>::GetRegister(std::string name, unsigned int index, VectorElement vectorElement) const
{
	return new IndexedRegister<T, V>(GetStructure(name), index, vectorElement); 
}

}
