#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Address/MemoryAddress.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface_Data(MoveAddressInstruction)

template<Bits B, class T, class S, bool Assert = true>
class MoveAddressInstruction : DispatchInherit(MoveAddressInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(MoveAddressInstruction,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(MoveAddressInstruction,
		REQUIRE_BASE(S, AddressableSpace)
	);

	MoveAddressInstruction(Register<PointerType<B, T, S>> *destination, MemoryAddress<B, T, S> *source)
		: m_destination(destination), m_source(source) {}

	// Properties

	const Register<PointerType<B, T, S>> *GetDestination() const { return m_destination; }
	Register<PointerType<B, T, S>> *GetDestination() { return m_destination; }
	void SetDestination(Register<PointerType<B, T, S>> *destination) { m_destination = destination; }

	const MemoryAddress<B, T, S> *GetAddress() const { return m_source; }
	MemoryAddress<B, T, S> *GetAddress() { return m_source; }
	void SetAddress(MemoryAddress<B, T, S> *address) { m_source = address; }

	// Formatting

	static std::string Mnemonic() { return "mov"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + PointerType<B, T, S>::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_source };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_source };
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Register<PointerType<B, T, S>> *m_destination = nullptr;
	MemoryAddress<B, T, S> *m_source = nullptr;
};

DispatchImplementation_Data(MoveAddressInstruction)

}
