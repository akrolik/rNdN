
#pragma once

#include "PTX/Operands/Register.h"

namespace PTX {

//T1 and T2 should be bit sizes of types
template<Bits A, class T1, class T2, VectorSize V = Scalar>
class ZeroExtendAddress : public Address<A, T1, V>
{
public:
	ZeroExtendAddress(Address<A, T2, V> *address) : m_address(address) {}

	std::string ToString()
	{
		return m_address->ToString();
	}

	std::string SpaceName()
	{
		//TODO: have the actual space name
		return ".global";
	}

private:
	Address<A, T2, V> *m_address = nullptr;
};

}
