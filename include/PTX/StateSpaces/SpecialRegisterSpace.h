#pragma once

#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class SpecialRegisterSpace : public RegisterSpace<T, V>
{
public:
	SpecialRegisterSpace(std::string prefix, unsigned int count) : RegisterSpace<T, V>(prefix, count) {}
	SpecialRegisterSpace(std::string name) : RegisterSpace<T, V>(name) {}
	SpecialRegisterSpace(std::vector<std::string> names) : RegisterSpace<T, V>(names) {}

	std::string SpaceName() { return ".sreg"; }
};

}
