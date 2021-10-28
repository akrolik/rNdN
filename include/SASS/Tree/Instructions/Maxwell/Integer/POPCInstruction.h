#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class POPCInstruction : public PredicatedInstruction
{
public:
	POPCInstruction(Register *destination, Register *source)
		: m_destination(destination), m_source(source) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Register *GetSource() const { return m_source; }
	Register *GetSource() { return m_source; }
	void SetSource(Register *source) { m_source = source; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_source };
	}

	// Formatting

	std::string OpCode() const override { return "POPC"; }

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString() + ", ";
		
		// Source
		code += m_source->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);

		return code;
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

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::SpecialFunction; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_source = nullptr;
};

}
