#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, Bits B, class S = AddressableSpace>
class PointerRegisterAdapter : public Register<PointerType<T, B, S>>
{ 
	REQUIRE_BASE_TYPE(PointerRegisterAdapter, DataType);
	REQUIRE_BASE_SPACE(PointerRegisterAdapter, AddressableSpace);
public:
	PointerRegisterAdapter(const Register<UIntType<B>> *variable) : Register<PointerType<T, B, S>>(variable->GetName()) {}
};

template<class T, class S = AddressableSpace>
using Pointer32RegisterAdapter = PointerRegisterAdapter<T, Bits::Bits32, S>;
template<class T, class S = AddressableSpace>
using Pointer64RegisterAdapter = PointerRegisterAdapter<T, Bits::Bits64, S>;

}
