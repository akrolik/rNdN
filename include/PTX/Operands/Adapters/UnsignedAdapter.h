#pragma once

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B>
class UnsignedRegisterAdapter : public Register<UIntType<B>>
{
public:
	UnsignedRegisterAdapter(const Register<IntType<B>> *variable) : Register<UIntType<B>>(variable->GetName()) {}
};

using Unsigned8RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits8>;
using Unsigned16RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits16>;
using Unsigned32RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits32>;
using Unsigned64RegisterAdapter = UnsignedRegisterAdapter<Bits::Bits64>;

}
