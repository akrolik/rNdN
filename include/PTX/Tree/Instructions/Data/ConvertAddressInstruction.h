#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface_Data(ConvertAddressInstruction)

template<Bits B, class T, class S, bool Assert = true>
class ConvertAddressInstruction : DispatchInherit(ConvertAddressInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(ConvertAddressInstruction,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(ConvertAddressInstruction,
		REQUIRE_BASE(S, AddressableSpace) && !REQUIRE_EXACT(S, AddressableSpace)
	);
	
	ConvertAddressInstruction(Register<PointerType<B, T>> *destination, Address<B, T, S> *source)
		: m_destination(destination), m_source(source) {}

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Properties

	const Register<PointerType<B, T>> *GetDestination() const { return m_destination; }
	Register<PointerType<B, T>> *GetDestination() { return m_destination; }
	void SetDestination(Register<PointerType<B, T>> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_source; }
	Address<B, T, S> *GetAddress() { return m_source; }
	void SetAddress(Address<B, T, S> *address) { m_source = address; } 

	// Formatting

	static std::string Mnemonic() { return "cvta"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + S::Name() + PointerType<B, T, S>::Name();
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

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Register<PointerType<B, T>> *m_destination = nullptr;
	Address<B, T, S> *m_source = nullptr;
};

template<class T, class S>
using ConvertAddress32Instruction = ConvertAddressInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using ConvertAddress64Instruction = ConvertAddressInstruction<Bits::Bits64, T, S>;

}
