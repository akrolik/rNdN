#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class Address : public TypedOperand<T>
{
	REQUIRE_TYPE_PARAM(Address,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(Address,
		REQUIRE_BASE(S, AddressableSpace)
	);
};

template<class T, class S>
using Address32 = Address<Bits::Bits32, T, S>;
template<class T, class S>
using Address64 = Address<Bits::Bits64, T, S>;

}
