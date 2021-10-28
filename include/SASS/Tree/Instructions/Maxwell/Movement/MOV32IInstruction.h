#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Immediate.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class MOV32IInstruction : public PredicatedInstruction
{
public:
	MOV32IInstruction(Register *destination, SizedImmediate<32> *source)
		: m_destination(destination), m_source(source) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const SizedImmediate<32> *GetSource() const { return m_source; }
	SizedImmediate<32> *GetSource() { return m_source; }
	void SetSource(SizedImmediate<32> *source) { m_source = source; }

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
	
	std::string OpCode() const override { return "MOV32I"; }

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// Source
		code += m_source->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x010000000000f000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return 0x0;
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandLiteral20W32(m_source);
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	SizedImmediate<32> *m_source = nullptr;
};

}
