#pragma once

#include "PTX/Tree/Operands/Address/Address.h"

namespace PTX {

template<Bits B, class D, class S, class SP>
class AddressAdapter : public Address<B, D, SP>
{
public:
	AddressAdapter(const Address<B, S, SP> *address) : m_address(address) {}

	std::string ToString() const { return m_address->ToString(); }

	Address<B, D, SP> *CreateOffsetAddress(int offset) const override
	{
		return new AddressAdapter<B, D, S, SP>(m_address->CreateOffsetAddress(offset));
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::AddressAdapter";
		j["destination"] = D::Name();
		j["source"] = S::Name();
		j["address"] = m_address->ToJSON();
		return j;
	}

protected:
	const Address<B, S, SP> *m_address = nullptr;
};

}
