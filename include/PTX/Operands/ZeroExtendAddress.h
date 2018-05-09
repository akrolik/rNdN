
#pragma once

#include "PTX/Operands/Register/Register.h"

namespace PTX {

//TODO: T1 and T2 should be bit sizes of types
template<Bits A, class T1, class T2, VectorSize V = Scalar>
class ZeroExtendAddress : public Address<A, T1, V>
{
public:
	ZeroExtendAddress(Address<A, T2, V> *address) : m_address(address) {}

	std::string ToString() const
	{
		return m_address->ToString();
	}

	AddressSpace GetSpace() const
	{
		return m_address->GetSpace();
	}

private:
	Address<A, T2, V> *m_address = nullptr;
};

}
