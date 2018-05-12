
#pragma once

#include "PTX/Operands/Address/Address.h"

namespace PTX {

template<Bits S, Bits D, class T>
class ZeroExtendAddress : public Address<D, T>
{
public:
	ZeroExtendAddress(Address<S, T> *address) : m_address(address) {}

	std::string ToString() const
	{
		return m_address->ToString();
	}

	AddressSpace GetSpace() const
	{
		return m_address->GetSpace();
	}

private:
	Address<S, T> *m_address = nullptr;
};

}
