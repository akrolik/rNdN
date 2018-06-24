#pragma once

#include "PTX/Operands/Adapters/RegisterAdapter.h"

namespace PTX {

template<Bits B>
class SignedRegisterAdapter : public RegisterAdapter<IntType<B>, UIntType<B>>
{
public:
	using RegisterAdapter<IntType<B>, UIntType<B>>::RegisterAdapter;

	json ToJSON() const override
	{
		json j = RegisterAdapter<IntType<B>, UIntType<B>>::ToJSON();
		j["kind"] = "PTX::SignedRegisterAdapter";
		return j;
	}
};

using Signed8RegisterAdapter = SignedRegisterAdapter<Bits::Bits8>;
using Signed16RegisterAdapter = SignedRegisterAdapter<Bits::Bits16>;
using Signed32RegisterAdapter = SignedRegisterAdapter<Bits::Bits32>;
using Signed64RegisterAdapter = SignedRegisterAdapter<Bits::Bits64>;

}
