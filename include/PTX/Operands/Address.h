#pragma once

#include "PTX/Operand.h"
#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class Address : public Operand<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	virtual std::string ToString() = 0;

	virtual std::string SpaceName() = 0;
};

}
