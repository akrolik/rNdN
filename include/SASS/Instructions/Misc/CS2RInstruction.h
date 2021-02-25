#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Register.h"
#include "SASS/Operands/SpecialRegister.h"

namespace SASS {

class CS2RInstruction : public PredicatedInstruction
{
public:
	CS2RInstruction(Register *destination, SpecialRegister *source)
		: PredicatedInstruction({destination, source}), m_destination(destination), m_source(source) {}

	// Properties
	
	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const SpecialRegister *GetSource() const { return m_source; }
	SpecialRegister *GetSource() { return m_source; }
	void SetSource(SpecialRegister *source) { m_source = source; }
	
	// Formatting

	std::string OpCode() const override { return "CS2R"; }
	std::string Operands() const override
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x50c8000000000000;
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandSpecialRegister(m_source);
	}

private:
	Register *m_destination = nullptr;
	SpecialRegister *m_source = nullptr;
};

}
