#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"

namespace PTX {

DispatchInterface_Data(PrefetchUniformInstruction)

template<Bits B, class T, class S = AddressableSpace, bool Assert = true>
class PrefetchUniformInstruction : DispatchInherit(PrefetchUniformInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(PrefetchUniformInstruction,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(PrefetchUniformInstruction,
		REQUIRE_EXACT(S, AddressableSpace)
	);

	PrefetchUniformInstruction(Address<B, T, AddressableSpace> *address) : m_address(address) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const Address<B, T, AddressableSpace> *GetAddress() const { return m_address; }
	Address<B, T, AddressableSpace> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, AddressableSpace> *address) { m_address = address; }

	// Formatting

	static std::string Mnemonic() { return "prefetch"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".L1";
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { new DereferencedAddress<B, T, AddressableSpace>(m_address) };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { new DereferencedAddress<B, T, AddressableSpace>(m_address) };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Address<B, T, AddressableSpace> *m_address = nullptr;
};

template<class T>
using PrefetchUniform32Instruction = PrefetchUniformInstruction<Bits::Bits32, T, AddressableSpace>;
template<class T>
using PrefetchUniform64Instruction = PrefetchUniformInstruction<Bits::Bits64, T, AddressableSpace>;

}
