#pragma once

#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<Bits B, class S>
class UnsignedVariableAdapter : public VariableAdapter<UIntType<B>, IntType<B>, S>
{
public:
	using VariableAdapter<UIntType<B>, IntType<B>, S>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<UIntType<B>, IntType<B>, S>::ToJSON();
		j["kind"] = "PTX::UnsignedVariableAdapter";
		return j;
	}
};

template<class S>
using Unsigned8VariableAdapter = UnsignedVariableAdapter<Bits::Bits8, S>;
template<class S>
using Unsigned16VariableAdapter = UnsignedVariableAdapter<Bits::Bits16, S>;
template<class S>
using Unsigned32VariableAdapter = UnsignedVariableAdapter<Bits::Bits32, S>;
template<class S>
using Unsigned64VariableAdapter = UnsignedVariableAdapter<Bits::Bits64, S>;

template<Bits B>
using UnsignedRegisterAdapter = UnsignedVariableAdapter<B, RegisterSpace>;

using Unsigned8RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits8>;
using Unsigned16RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits16>;
using Unsigned32RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits32>;
using Unsigned64RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits64>;

}
