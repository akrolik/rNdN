#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface_Data(IsSpaceInstruction)

template<Bits B, class T, class S, bool Assert = true>
class IsSpaceInstruction : DispatchInherit(IsSpaceInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(IsSpaceInstruction,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(IsSpaceInstruction,
		REQUIRE_BASE(S, AddressableSpace) && !REQUIRE_EXACT(S, AddressableSpace)
	);

	IsSpaceInstruction(Register<PredicateType> *destination, Address<B, T> *address) : m_destination(destination), m_address(address) {}

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Properties

	const Register<PredicateType> *GetDestination() const { return m_destination; }
	Register<PredicateType> *GetDestination() { return m_destination; }
	void SetDestination(Register<PredicateType> *destination) { m_destination = destination; }

	const Address<B, T> *GetAddress() const { return m_address; }
	Address<B, T> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T> *address) { m_address = address; }

	// Formatting

	static std::string Mnemonic() { return "isspacep"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + S::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_address };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_address };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Register<PredicateType> *m_destination = nullptr;
	Address<B, T> *m_address = nullptr;
};

template<class T, class S>
using IsSpace32Instruction = IsSpaceInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using IsSpace64Instruction = IsSpaceInstruction<Bits::Bits64, T, S>;

}
