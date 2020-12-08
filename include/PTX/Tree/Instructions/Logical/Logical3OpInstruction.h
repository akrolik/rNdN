#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Tree/Operands/Extended/HexOperand.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(Logical3OpInstruction)

template<class T = Bit32Type, bool Assert = true>
class Logical3OpInstruction : DispatchInherit(Logical3OpInstruction), public InstructionBase_3<T>
{
public:
	REQUIRE_TYPE_PARAM(Logical3OpInstruction,
		REQUIRE_EXACT(T, Bit32Type)
	);

	Logical3OpInstruction(const Register<T> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, const TypedOperand<T> *sourceC, std::uint8_t immLut) : InstructionBase_3<T>(destination, sourceA, sourceB, sourceC), m_immLut(immLut) {}

	std::uint8_t GetLookup() const { return m_immLut; }
	void SetLookup(std::uint8_t lookup) { m_immLut = lookup; }

	static std::string Mnemonic() { return "lop3"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		auto operands = InstructionBase_3<T>::Operands();
		operands.push_back(new HexOperand(m_immLut));
		return operands;
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	// @uint8_t m_immLut
	//
	// Computed look-up-table value for the 3 part boolean operation
	//
	// Take ta = 0xF0
	//      tb = 0xCC
	//      tc = 0xAA
	// and perform the computation you expect
	//
	// i.e. To perform the operation
	//            a & b & c
	// compute the value of
	//        0xF0 & 0xCC & 0xAA = 0x80

	std::uint8_t m_immLut = 0x00;
};

DispatchImplementation(Logical3OpInstruction)

}
