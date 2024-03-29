#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface_Data(LoadUniformInstruction)

template<Bits B, class T, class S, bool Assert = true>
class LoadUniformInstruction : DispatchInherit(LoadUniformInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(LoadUniformInstruction,
		REQUIRE_BASE(T, ValueType) && !REQUIRE_EXACT(T,
			Float16Type, Float16x2Type,
			Vector2Type<Float16Type>, Vector2Type<Float16x2Type>,
			Vector4Type<Float16Type>, Vector4Type<Float16x2Type>
		)
	);
	REQUIRE_SPACE_PARAM(LoadUniformInstruction,
		REQUIRE_EXACT(S, AddressableSpace, GlobalSpace)
	);

	LoadUniformInstruction(Register<T> *destination, Address<B, T, S> *address) : m_destination(destination), m_address(address) {}

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Properties

	const Register<T> *GetDestination() const { return m_destination; }
	Register<T> *GetDestination() { return m_destination; }
	void SetDestination(Register<T> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_address; }
	Address<B, T, S> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, S> *address) { m_address = address; }

	// Formatting

	static std::string Mnemonic() { return "ldu"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + S::Name() + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, new DereferencedAddress<B, T, S>(m_address) };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, new DereferencedAddress<B, T, S>(m_address) };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Register<T> *m_destination = nullptr;
	Address<B, T, S> *m_address = nullptr;
};

template<class T, class S>
using LoadUniform32Instruction = LoadUniformInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using LoadUniform64Instruction = LoadUniformInstruction<Bits::Bits64, T, S>;

}
