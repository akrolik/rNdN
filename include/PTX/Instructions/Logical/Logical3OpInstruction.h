#pragma once

#include "PTX/Instructions/InstructionBase.h"

#include "PTX/Operands/Extended/StringOperand.h"

namespace PTX {

class Logical3OpInstruction : public InstructionBase_3<Bit32Type>
{
public:
	Logical3OpInstruction(const Register<Bit32Type> *destination, const TypedOperand<Bit32Type> *sourceA, const TypedOperand<Bit32Type> *sourceB, const TypedOperand<Bit32Type> *sourceC, unsigned char immLut) : InstructionBase_3<Bit32Type>(destination, sourceA, sourceB, sourceC), m_immLut(immLut) {}

	std::string OpCode() const override
	{
		return "lop3" + Bit32Type::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		auto operands = InstructionBase_3<Bit32Type>::Operands();
		std::ostringstream hex;
		hex << std::hex << static_cast<int>(m_immLut);
		operands.push_back(new StringOperand(std::string("0x") + hex.str()));
		return operands;
	}

private:
	// @unsigned char m_immLut
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

	unsigned char m_immLut;
};

}
