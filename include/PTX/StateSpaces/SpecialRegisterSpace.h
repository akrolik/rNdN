#pragma once

#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class SpecialRegisterSpace : public RegisterSpace<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	using RegisterSpace<T, V>::RegisterSpace;

	std::string Specifier() const { return ".sreg"; }
};

}
