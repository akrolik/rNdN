#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Address/RegisterAddress.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface_Data(ConvertToAddressInstruction)

template<Bits B, class T, class S, bool Assert = true>
class ConvertToAddressInstruction : DispatchInherit(ConvertToAddressInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(ConvertToAddressInstruction,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(ConvertToAddressInstruction,
		REQUIRE_BASE(S, AddressableSpace) && !REQUIRE_EXACT(S, AddressableSpace)
	);
	
	ConvertToAddressInstruction(Register<PointerType<B, T, S>> *destination, RegisterAddress<B, T> *source)
		: m_destination(destination), m_source(source) {}

	// Properties

	const Register<PointerType<B, T, S>> *GetDestination() const { return m_destination; }
	Register<PointerType<B, T, S>> *GetDestination() { return m_destination; }
	void SetDestination(Register<PointerType<B, T, S>> *destination) { m_destination = destination; }

	const RegisterAddress<B, T> *GetAddress() const { return m_source; }
	RegisterAddress<B, T> *GetAddress() { return m_source; }
	void SetAddress(RegisterAddress<B, T> *address) { m_source = address; } 

	// Formatting

	static std::string Mnemonic() { return "cvta.to"; }

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

	Register<PointerType<B, T, S>> *m_destination = nullptr;
	RegisterAddress<B, T> *m_source = nullptr;
};

template<class T, class S>
using ConvertToAddress32Instruction = ConvertToAddressInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using ConvertToAddress64Instruction = ConvertToAddressInstruction<Bits::Bits64, T, S>;

}
