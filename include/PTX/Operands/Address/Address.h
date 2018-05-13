#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<Bits A, class T, AddressSpace S>
class Address : public Operand<T>
{
};

template<class T, AddressSpace S>
using Address32 = Address<Bits::Bits32, T, S>;
template<class T, AddressSpace S>
using Address64 = Address<Bits::Bits64, T, S>;

}
