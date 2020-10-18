#pragma once

#include "SASS/Instructions/Instruction.h"
#include "SASS/Operands/Operand.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class MOVInstruction : public Instruction
{
public:
	MOVInstruction(const Register *destination, const Operand *source) : Instruction({destination, source}), m_destination(destination), m_source(source) {}
	
	std::string OpCode() const override { return "MOV"; }

private:
	const Register *m_destination = nullptr;
	const Operand *m_source = nullptr;
};

}
