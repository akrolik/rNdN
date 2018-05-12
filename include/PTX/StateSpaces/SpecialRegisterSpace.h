#pragma once

#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class T>
class SpecialRegisterSpace : public RegisterSpace<T>
{
public:
	using RegisterSpace<T>::RegisterSpace;

	std::string Specifier() const { return ".sreg"; }
};

}
