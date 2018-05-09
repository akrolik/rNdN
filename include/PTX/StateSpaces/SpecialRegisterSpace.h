#pragma once

#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class SpecialRegisterSpace : public RegisterSpace<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	SpecialRegisterSpace(std::string prefix, unsigned int count) : RegisterSpace<T, V>(prefix, count) {}
	SpecialRegisterSpace(std::string name) : RegisterSpace<T, V>(name) {}
	SpecialRegisterSpace(std::vector<std::string> names) : RegisterSpace<T, V>(names) {}

	std::string Specifier() const { return ".sreg"; }
};

}
