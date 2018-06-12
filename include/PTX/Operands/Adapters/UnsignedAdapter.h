#pragma once

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<Bits B>
class UnsignedAdapter : public Register<UIntType<B>>
{
public:
	UnsignedAdapter(const Register<IntType<B>> *variable) : Register<UIntType<B>>(variable->GetName()) {}
};

using Unsigned32Adapter = UnsignedAdapter<Bits::Bits32>;
using Unsigned64Adapter = UnsignedAdapter<Bits::Bits64>;

}
