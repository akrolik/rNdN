#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

class Logical3OpInstruction : public InstructionBase<Bit32Type, 3>
{
public:
	Logical3OpInstruction(Register<Bit32Type> *destination, Operand<Bit32Type> *sourceA, Operand<Bit32Type> *sourceB, Operand<Bit32Type> *sourceC, unsigned char immLut) : InstructionBase<Bit32Type, 3>(destination, sourceA, sourceB, sourceC), m_immLut(immLut) {}

	std::string OpCode() const
	{
		return "lop3" + Bit32Type::Name();
	}

	std::string Operands() const
	{
		std::ostringstream hex;
		hex << std::hex << static_cast<int>(m_immLut);
		return InstructionBase<Bit32Type, 3>::Operands() + ", 0x" + hex.str();
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
