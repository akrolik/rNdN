#pragma once

#include "PTX/Tree/Operands/Operand.h"

#include "PTX/Tree/Operands/Address/Address.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class DereferencedAddress : public TypedOperand<T>
{
public:
	REQUIRE_TYPE_PARAM(DereferencedAddress,
		REQUIRE_BASE(T, Type)
	);
	REQUIRE_SPACE_PARAM(DereferencedAddress,
		REQUIRE_BASE(S, AddressableSpace)
	);

	DereferencedAddress(const Address<B, T, S> *address) : m_address(address) {}

	std::string ToString() const override
	{
		return "[" + m_address->ToString() + "]";
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::DereferencedAddress";
		j["address"] = m_address->ToJSON();
		return j;
	}

private:
	const Address<B, T, S> *m_address = nullptr;
};

template<class T, class S>
using DereferencedAddress32 = DereferencedAddress<Bits::Bits32, T, S>;
template<class T, class S>
using DereferencedAddress64 = DereferencedAddress<Bits::Bits64, T, S>;

}
