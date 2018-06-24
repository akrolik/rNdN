#pragma once

#include "PTX/Operands/Adapters/RegisterAdapter.h"

namespace PTX {

template<Bits B>
class UnsignedRegisterAdapter : public RegisterAdapter<UIntType<B>, IntType<B>>
{
public:
	using RegisterAdapter<UIntType<B>, IntType<B>>::RegisterAdapter;

	json ToJSON() const override
	{
		json j = RegisterAdapter<UIntType<B>, IntType<B>>::ToJSON();
		j["kind"] = "PTX::UnsignedRegisterAdapter";
		return j;
	}
};

using Unsigned8RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits8>;
using Unsigned16RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits16>;
using Unsigned32RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits32>;
using Unsigned64RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits64>;

}
