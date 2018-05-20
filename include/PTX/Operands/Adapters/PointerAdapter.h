#pragma once

#include "PTX/Operands/Variables/Register.h"

#include "PTX/StateSpaces/SpaceAdapter.h"

namespace PTX {

template<class T, Bits B, AddressSpace A = AddressSpace::Generic>
class PointerAdapter : public Register<PointerType<T, B, A>>
{
public:
	PointerAdapter(Register<UIntType<B>> *variable) : Register<PointerType<T, B, A>>(variable->GetName(), new RegisterSpaceAdapter<PointerType<T, B, A>, UIntType<B>>(variable->GetStateSpace())) {}
};

template<class T, AddressSpace A = AddressSpace::Generic>
using Pointer32Adapter = PointerAdapter<T, Bits::Bits32, A>;
template<class T, AddressSpace A = AddressSpace::Generic>
using Pointer64Adapter = PointerAdapter<T, Bits::Bits64, A>;

}
