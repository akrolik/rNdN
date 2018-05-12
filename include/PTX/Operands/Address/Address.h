#pragma once

#include "PTX/Operands/Operand.h"
#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<Bits A, class T>
class Address : public Operand<T>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	virtual AddressSpace GetSpace() const = 0;
};

template<class T>
using Address32 = Address<Bits::Bits32, T>;
template<class T>
using Address64 = Address<Bits::Bits64, T>;

}
