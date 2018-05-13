#pragma once

#include "PTX/Operands/Variable.h"
#include "PTX/StateSpaces/SpaceAdapter.h"

namespace PTX {

template<Bits B>
class UnsignedAdapter : public Register<UIntType<B>>
{
public:
	UnsignedAdapter(Register<IntType<B>> *variable) : Register<UIntType<B>>(variable->GetName(), new RegisterSpaceAdapter<UIntType<B>, IntType<B>>(variable->GetStateSpace())) {}
};

using Unsigned32Adapter = UnsignedAdapter<Bits::Bits32>;
using Unsigned64Adapter = UnsignedAdapter<Bits::Bits64>;

}
