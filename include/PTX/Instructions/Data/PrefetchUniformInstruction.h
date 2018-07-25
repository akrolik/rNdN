#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"

namespace PTX {

template<Bits B, class T, bool Assert = true>
class PrefetchUniformInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(PrefetchUniformInstruction,
		REQUIRE_BASE(T, ValueType)
	);

	PrefetchUniformInstruction(const Address<B, T, AddressableSpace> *address) : m_address(address) {}

	const Address<B, T, AddressableSpace> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, AddressableSpace> *address) { m_address = address; }

	static std::string Mnemonic() { return "prefetch"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".L1";
	}

	std::vector<const Operand *> Operands() const override
	{
		return { new DereferencedAddress<B, T, AddressableSpace>(m_address) };
	}

private:
	const Address<B, T, AddressableSpace> *m_address = nullptr;
};

template<class T>
using PrefetchUniform32Instruction = PrefetchUniformInstruction<Bits::Bits32, T>;
template<class T>
using PrefetchUniform64Instruction = PrefetchUniformInstruction<Bits::Bits64, T>;

}
