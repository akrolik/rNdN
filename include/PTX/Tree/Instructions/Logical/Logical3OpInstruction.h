#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Tree/Operands/Constants/Value.h"

namespace PTX {

DispatchInterface(Logical3OpInstruction)

template<class T = Bit32Type, bool Assert = true>
class Logical3OpInstruction : DispatchInherit(Logical3OpInstruction), public InstructionBase_3<T>
{
public:
	REQUIRE_TYPE_PARAM(Logical3OpInstruction,
		REQUIRE_EXACT(T, Bit32Type)
	);

	Logical3OpInstruction(Register<T> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, TypedOperand<T> *sourceC, UInt8Value *immLut)
		: InstructionBase_3<T>(destination, sourceA, sourceB, sourceC), m_immLut(immLut) {}

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Properties

	const UInt8Value *GetLookup() const { return m_immLut; }
	UInt8Value *GetLookup() { return m_immLut; }
	void SetLookup(UInt8Value *lookup) { m_immLut = lookup; }
	
	// Formatting

	static std::string Mnemonic() { return "lop3"; }

	std::string GetGetOpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		auto operands = InstructionBase_3<T>::Operands();
		operands.push_back(m_immLut);
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		auto operands = InstructionBase_3<T>::Operands();
		operands.push_back(m_immLut);
		return operands;
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
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

	UInt8Value *m_immLut = nullptr;
};

}
