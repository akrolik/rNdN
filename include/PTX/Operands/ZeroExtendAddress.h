
#pragma once

#include "PTX/Operands/Address/Address.h"

namespace PTX {

template<Bits S, Bits D, class T, AddressSpace SP>
class ZeroExtendAddress : public Address<D, T, SP>
{
public:
	ZeroExtendAddress(Address<S, T, SP> *address) : m_address(address) {}

	std::string ToString() const
	{
		return m_address->ToString();
	}

private:
	Address<S, T, SP> *m_address = nullptr;
};

}
