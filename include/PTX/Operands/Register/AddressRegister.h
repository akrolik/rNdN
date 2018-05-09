#pragma once

#include "PTX/Operands/Register/Register.h"
#include "PTX/Type.h"

namespace PTX {

template<class T>
class AddressRegister : public Register<T, Scalar>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	AddressRegister(typename RegisterSpace<T>::Element *element, unsigned int index = 0, AddressSpace addressSpace = AddressSpace::Generic) : Register<T, Scalar>(element, index), m_addressSpace(addressSpace) {}

	virtual AddressSpace GetAddressSpace() const { return m_addressSpace; }

private:
	AddressSpace m_addressSpace;
};

}
