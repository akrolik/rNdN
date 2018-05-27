#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<Bits B, class T, AddressSpace A = AddressSpace::Generic>
class Address : public Operand<T>
{
};

template<class T, AddressSpace A>
using Address32 = Address<Bits::Bits32, T, A>;
template<class T, AddressSpace A>
using Address64 = Address<Bits::Bits64, T, A>;

}
