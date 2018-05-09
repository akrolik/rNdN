#pragma once

#include "PTX/Operands/Operand.h"
#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class Address : public Operand<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	virtual AddressSpace GetSpace() const = 0;
};

}
