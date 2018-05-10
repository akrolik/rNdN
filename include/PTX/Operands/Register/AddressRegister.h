#pragma once

#include "PTX/Operands/Register/Register.h"
#include "PTX/Type.h"

namespace PTX {

template<Bits A>
class AddressRegister : public Register<UIntType<A>, Scalar>
{
public:
	AddressRegister(Register<UIntType<A>, Scalar> *reg, AddressSpace addressSpace = AddressSpace::Generic) : Register<UIntType<A>, Scalar>(reg->m_structure, reg->m_index), m_addressSpace(addressSpace) {}

	virtual AddressSpace GetAddressSpace() const { return m_addressSpace; }

private:
	AddressSpace m_addressSpace;
};

}
