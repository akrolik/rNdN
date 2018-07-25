#pragma once

#include "PTX/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<Bits B, class S>
class SignedVariableAdapter : public VariableAdapter<IntType<B>, UIntType<B>, S>
{
public:
	using VariableAdapter<IntType<B>, UIntType<B>, S>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<IntType<B>, UIntType<B>, S>::ToJSON();
		j["kind"] = "PTX::SignedVariableAdapter";
		return j;
	}
};

template<class S>
using Signed8VariableAdapter = SignedVariableAdapter<Bits::Bits8, S>;
template<class S>
using Signed16VariableAdapter = SignedVariableAdapter<Bits::Bits16, S>;
template<class S>
using Signed32VariableAdapter = SignedVariableAdapter<Bits::Bits32, S>;
template<class S>
using Signed64VariableAdapter = SignedVariableAdapter<Bits::Bits64, S>;

template<Bits B>
using SignedRegisterAdapter = SignedVariableAdapter<B, RegisterSpace>;

using Signed8RegisterAdapter = SignedRegisterAdapter<Bits::Bits8>;
using Signed16RegisterAdapter = SignedRegisterAdapter<Bits::Bits16>;
using Signed32RegisterAdapter = SignedRegisterAdapter<Bits::Bits32>;
using Signed64RegisterAdapter = SignedRegisterAdapter<Bits::Bits64>;

}
