#pragma once

#include "PTX/Type.h"

namespace PTX {

template<class T>
class Operand
{
	static_assert(std::is_base_of<ValueType, T>::value, "T must be a PTX::ValueType");
public:
	virtual std::string ToString() const = 0;
};

}
