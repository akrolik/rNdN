#pragma once

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B>
class SignedAdapter : public Register<IntType<B>>
{
public:
	SignedAdapter(const Register<UIntType<B>> *variable) : Register<IntType<B>>(variable->GetName()) {}
};

using Signed32Adapter = SignedAdapter<Bits::Bits32>;
using Signed64Adapter = SignedAdapter<Bits::Bits64>;

}
