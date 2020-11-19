#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class POPCInstruction : public PredicatedInstruction
{
public:
	POPCInstruction(const Register *destination, const Register *source)
		: PredicatedInstruction({destination, source}), m_destination(destination), m_source(source) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	void SetDestination(const Register *destination) { m_destination = destination; }

	const Register *GetSource() const { return m_source; }
	void SetSource(const Register *source) { m_source = source; }

	// Formatting

	std::string OpCode() const override { return "POPC"; }

	std::string Operands() const override
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x5c08000000000000;
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister20(m_source);
	}

private:
	const Register *m_destination = nullptr;
	const Register *m_source = nullptr;
};

}
