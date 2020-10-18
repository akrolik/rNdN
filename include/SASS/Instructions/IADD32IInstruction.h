#pragma once

#include "SASS/Instructions/Instruction.h"
#include "SASS/Operands/Immediate.h"
#include "SASS/Operands/Operand.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class IADD32IInstruction : public Instruction
{
public:
	IADD32IInstruction(const Register *destination, const Operand *sourceA, const Immediate *sourceB) : Instruction({destination, sourceA, sourceB}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string OpCode() const override { return "IADD32I"; }

private:
	const Register *m_destination = nullptr;
	const Operand *m_sourceA = nullptr;
	const Immediate *m_sourceB = nullptr;
};

}
