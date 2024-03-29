#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace, bool Assert = true>
class Address : public Operand
{
public:
	REQUIRE_TYPE_PARAM(Address,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(Address,
		REQUIRE_BASE(S, AddressableSpace)
	);

	virtual Address<B, T, S, Assert> *CreateOffsetAddress(int offset) const = 0;
};

template<class T, class S>
using Address32 = Address<Bits::Bits32, T, S>;
template<class T, class S>
using Address64 = Address<Bits::Bits64, T, S>;

}
