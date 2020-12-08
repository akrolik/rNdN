#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	const Address<B, T, AddressableSpace> *m_address = nullptr;
};

DispatchImplementation_Data(PrefetchUniformInstruction)

template<class T>
using PrefetchUniform32Instruction = PrefetchUniformInstruction<Bits::Bits32, T, AddressableSpace>;
template<class T>
using PrefetchUniform64Instruction = PrefetchUniformInstruction<Bits::Bits64, T, AddressableSpace>;

}
