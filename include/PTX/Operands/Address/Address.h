#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class Address : public Operand<T>
{
	REQUIRE_BASE_TYPE(Address, DataType);
	REQUIRE_BASE_SPACE(Address, AddressableSpace);
};

template<class T, class S>
using Address32 = Address<Bits::Bits32, T, S>;
template<class T, class S>
using Address64 = Address<Bits::Bits64, T, S>;

}
