#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, Bits B, class S = AddressableSpace>
class PointerAdapter : public Register<PointerType<T, B, S>>
{ 
	REQUIRE_BASE_TYPE(PointerAdapter, Type);
	REQUIRE_BASE_SPACE(PointerAdapter, AddressableSpace);
public:
	PointerAdapter(const Register<UIntType<B>> *variable) : Register<PointerType<T, B, S>>(variable->GetName()) {}
};

template<class T, class S = AddressableSpace>
using Pointer32Adapter = PointerAdapter<T, Bits::Bits32, S>;
template<class T, class S = AddressableSpace>
using Pointer64Adapter = PointerAdapter<T, Bits::Bits64, S>;

}
