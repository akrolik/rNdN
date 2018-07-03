#pragma once

#include "PTX/Operands/Operand.h"

#include "PTX/Operands/Address/Address.h"

namespace PTX {

template<Bits B, class T, class S = AddressableSpace>
class DereferenceOperand : public TypedOperand<T>
{
public:
	REQUIRE_TYPE_PARAM(DereferenceOperand,
		REQUIRE_BASE(T, Type)
	);
	REQUIRE_SPACE_PARAM(DereferenceOperand,
		REQUIRE_BASE(S, AddressableSpace)
	);

	DereferenceOperand(const Address<B, T, S> *address) : m_address(address) {}

	std::string ToString() const override
	{
		return "[" + m_address->ToString() + "]";
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::DereferenceOperand";
		j["address"] = m_address->ToJSON();
		return j;
	}

private:
	const Address<B, T, S> *m_address = nullptr;
};

template<class T, class S>
using Dereference32Operand = DereferenceOperand<Bits::Bits32, T, S>;
template<class T, class S>
using Dereference64Operand = DereferenceOperand<Bits::Bits64, T, S>;

}
