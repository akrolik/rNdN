#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface_Data(IsSpaceInstruction)

template<Bits B, class T, class S, bool Assert = true>
class IsSpaceInstruction : DispatchInherit(IsSpaceInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(IsSpaceInstruction,
		REQUIRE_BASE(T, Type)
	);
	REQUIRE_SPACE_PARAM(IsSpaceInstruction,
		REQUIRE_BASE(S, AddressableSpace)
	);

	IsSpaceInstruction(const Register<PredicateType> *destination, const Address<B, T> *address) : m_destination(destination), m_address(address) {}

	const Register<PredicateType> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<PredicateType> *destination) { m_destination = destination; }

	const Address<B, T> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T> *address) { m_address = address; }

	static std::string Mnemonic() { return "isspacep"; }

	std::string OpCode() const override
	{
		return Mnemonic() + S::Name();
	}

	std::string Operands() const override
	{
		return m_destination->ToString() + ", " + m_address->ToString();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	const Register<PredicateType> *m_destination = nullptr;
	const Address<B, T> *m_address = nullptr;
};

DispatchImplementation_Data(IsSpaceInstruction);

template<class T, class S>
using IsSpace32Instruction = IsSpaceInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using IsSpace64Instruction = IsSpaceInstruction<Bits::Bits64, T, S>;

}
