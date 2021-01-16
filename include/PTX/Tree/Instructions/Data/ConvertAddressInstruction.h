#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

DispatchInterface_Data2(ConvertAddressInstruction)

template<Bits B, class T, class D, class S, bool Assert = true>
class ConvertAddressInstruction : DispatchInherit(ConvertAddressInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(ConvertAddressInstruction,
		REQUIRE_BASE(T, Type)
	);
	REQUIRE_SPACE_PARAMS(ConvertAddressInstruction,
		REQUIRE_BASE(D, AddressableSpace),
		REQUIRE_BASE(S, AddressableSpace)
	);
	
	ConvertAddressInstruction(Register<PointerType<B, T, D>> *destination, Address<B, T, S> *source)
		: m_destination(destination), m_source(source) {}

	// Properties

	const Register<PointerType<B, T, D>> *GetDestination() const { return m_destination; }
	Register<PointerType<B, T, D>> *GetDestination() { return m_destination; }
	void SetDestination(Register<PointerType<B, T, D>> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_source; }
	Address<B, T, S> *GetAddress() { return m_source; }
	void SetAddress(Address<B, T, S> *address) { m_source = address; } 

	// Formatting

	static std::string Mnemonic() { return "cvta"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(std::is_same<S, AddressableSpace>::value)
		{
			code += ".to" + D::Name();
		}
		else
		{
			code += S::Name();
		}
		return code + PointerType<B, T, D>::Name();
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
	DispatchMember_Space1(D);
	DispatchMember_Space2(S);

	Register<PointerType<B, T, D>> *m_destination = nullptr;
	Address<B, T, S> *m_source = nullptr;
};

template<class T, class D, class S>
using ConvertAddress32Instruction = ConvertAddressInstruction<Bits::Bits32, T, D, S>;
template<class T, class D, class S>
using ConvertAddress64Instruction = ConvertAddressInstruction<Bits::Bits64, T, D, S>;

template<Bits B, class T, class D>
using ConvertToAddressInstruction = ConvertAddressInstruction<B, T, D, AddressableSpace>;
template<class T, class D>
using ConvertToAddress32Instruction = ConvertToAddressInstruction<Bits::Bits32, T, D>;
template<class T, class D>
using ConvertToAddress64Instruction = ConvertToAddressInstruction<Bits::Bits64, T, D>;

}
