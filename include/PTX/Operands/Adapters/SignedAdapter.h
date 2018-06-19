#pragma once

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B>
class SignedRegisterAdapter : public Register<IntType<B>>
{
public:
	SignedRegisterAdapter(const Register<UIntType<B>> *variable) : Register<IntType<B>>(variable->GetName()) {}
};

using Signed8RegisterAdapter = SignedRegisterAdapter<Bits::Bits8>;
using Signed16RegisterAdapter = SignedRegisterAdapter<Bits::Bits16>;
using Signed32RegisterAdapter = SignedRegisterAdapter<Bits::Bits32>;
using Signed64RegisterAdapter = SignedRegisterAdapter<Bits::Bits64>;

}
