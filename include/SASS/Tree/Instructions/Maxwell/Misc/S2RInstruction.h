#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/SpecialRegister.h"

namespace SASS {
namespace Maxwell {

class S2RInstruction : public PredicatedInstruction
{
public:
	S2RInstruction(Register *destination, SpecialRegister *source)
		: m_destination(destination), m_source(source) {}

	// Properties
	
	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const SpecialRegister *GetSource() const { return m_source; }
	SpecialRegister *GetSource() { return m_source; }
	void SetSource(SpecialRegister *source) { m_source = source; }
	
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

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::S2R; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	SpecialRegister *m_source = nullptr;
};

}
}
