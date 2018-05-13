#pragma once

#include "PTX/Operands/Variables/Register.h"

#include "PTX/StateSpaces/SpaceAdapter.h"

namespace PTX {

template<Bits B>
class SignedAdapter : public Register<IntType<B>>
{
public:
	SignedAdapter(Register<UIntType<B>> *variable) : Register<IntType<B>>(variable->GetName(), new RegisterSpaceAdapter<IntType<B>, UIntType<B>>(variable->GetStateSpace())) {}
};

using Signed32Adapter = SignedAdapter<Bits::Bits32>;
using Signed64Adapter = SignedAdapter<Bits::Bits64>;

}
