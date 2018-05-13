#pragma once

#include "PTX/Type.h"

namespace PTX {

template<class T>
class Operand
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	virtual std::string ToString() const = 0;
};

}
