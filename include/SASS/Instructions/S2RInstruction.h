#pragma once

#include "SASS/Instructions/Instruction.h"
#include "SASS/Operands/Register.h"
#include "SASS/Operands/SpecialRegister.h"

namespace SASS {

class S2RInstruction : public Instruction
{
public:
	S2RInstruction(const Register *destination, const SpecialRegister *source) : Instruction({destination, source}), m_destination(destination), m_source(source) {}

	std::string OpCode() const override { return "S2R"; }

private:
	const Register *m_destination = nullptr;
	const SpecialRegister *m_source = nullptr;
};

}
