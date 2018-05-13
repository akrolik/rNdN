#pragma once

#include "PTX/Type.h"
#include "PTX/Operands/Variables/Register.h"
#include "PTX/StateSpaces/AddressableSpace.h"

namespace PTX {

template<Bits A, class T, AddressSpace S>
class AddressRegister//TODO: : public Register<UIntType<A>>
{
public:
	AddressRegister(Register<UIntType<A>> *variable, AddressableSpace<T, S> *addressableSpace) : m_variable(variable), m_addressableSpace(addressableSpace) {}

	std::string ToString() const
	{
		return m_variable->ToString();
	}

private:
	Register<UIntType<A>> *m_variable;
	AddressableSpace<T, S> *m_addressableSpace;
};

template<class T, AddressSpace S>
using Address32Register = AddressRegister<Bits::Bits32, T, S>;
template<class T, AddressSpace S>
using Address64Register = AddressRegister<Bits::Bits64, T, S>;

}
