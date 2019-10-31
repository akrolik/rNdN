#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class Address : public TypedOperand<T>
{
public:
	REQUIRE_TYPE_PARAM(Address,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(Address,
		REQUIRE_BASE(S, AddressableSpace)
	);

	virtual Address<B, T, S> *CreateOffsetAddress(int offset) const = 0;
};

template<class T, class S>
using Address32 = Address<Bits::Bits32, T, S>;
template<class T, class S>
using Address64 = Address<Bits::Bits64, T, S>;

}
