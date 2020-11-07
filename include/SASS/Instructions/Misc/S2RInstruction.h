#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Register.h"
#include "SASS/Operands/SpecialRegister.h"

namespace SASS {

class S2RInstruction : public Instruction
{
public:
	S2RInstruction(const Register *destination, const SpecialRegister *source)
		: Instruction({destination, source}), m_destination(destination), m_source(source) {}

	// Properties
	
	const Register *GetDestination() const { return m_destination; }
	void SetDestination(const Register *destination) { m_destination = destination; }

	const SpecialRegister *GetSource() const { return m_source; }
	void SetSource(const SpecialRegister *source) { m_source = source; }
	
	// Formatting

	std::string OpCode() const override { return "S2R"; }
	std::string Operands() const override
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xf0c8000000000000;
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandSpecialRegister(m_source);
	}

private:
	const Register *m_destination = nullptr;
	const SpecialRegister *m_source = nullptr;
};

}
