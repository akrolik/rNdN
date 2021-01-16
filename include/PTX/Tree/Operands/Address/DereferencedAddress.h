#pragma once

#include "PTX/Tree/Operands/Operand.h"

#include "PTX/Tree/Operands/Address/Address.h"

namespace PTX {

DispatchInterface_Data(DereferencedAddress)

template<Bits B, class T, class S = AddressableSpace, bool Assert = true>
class DereferencedAddress : DispatchInherit(DereferencedAddress), public TypedOperand<T, Assert>
{
public:
	REQUIRE_TYPE_PARAM(DereferencedAddress,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(DereferencedAddress,
		REQUIRE_BASE(S, AddressableSpace)
	);

	DereferencedAddress(Address<B, T, S> *address) : m_address(address) {}

	// Properties

	const Address<B, T, S> *GetAddress() const { return m_address; }
	Address<B, T, S> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, S> *address) { m_address = address; }

	// Formatting

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

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Address<B, T, S> *m_address = nullptr;
};

DispatchImplementation_Data(DereferencedAddress)

template<class T, class S>
using DereferencedAddress32 = DereferencedAddress<Bits::Bits32, T, S>;
template<class T, class S>
using DereferencedAddress64 = DereferencedAddress<Bits::Bits64, T, S>;

}
