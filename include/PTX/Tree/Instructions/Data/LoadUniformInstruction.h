#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	LoadUniformInstruction(const Register<T> *destination, const Address<B, T, S> *address) : m_destination(destination), m_address(address) {}

	const Register<T> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<T> *destination) { m_destination = destination; }

	const Address<B, T, S> *GetAddress() const { return m_address; }
	void SetAddress(const Address<B, T, S> *address) { m_address = address; }

	static std::string Mnemonic() { return "ldu"; }

	std::string OpCode() const override
	{
		return Mnemonic() + S::Name() + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, new DereferencedAddress<B, T, S>(m_address) };
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	const Register<T> *m_destination = nullptr;
	const Address<B, T, S> *m_address = nullptr;
};

DispatchImplementation_Data(LoadUniformInstruction)

template<class T, class S>
using LoadUniform32Instruction = LoadUniformInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using LoadUniform64Instruction = LoadUniformInstruction<Bits::Bits64, T, S>;

}
