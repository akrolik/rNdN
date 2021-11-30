#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class MOVInstruction : public PredicatedInstruction
{
public:
	MOVInstruction(Register *destination, Composite *sourceA, I8Immediate *sourceB = nullptr)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Composite *GetSourceA() const { return m_sourceA; }
	Composite *GetSourceA() { return m_sourceA; }
	void SetSourceA(Composite *sourceA) { m_sourceA = sourceA; }

	const I8Immediate *GetSourceB() const { return m_sourceB; }
	I8Immediate *GetSourceB() { return m_sourceB; }
	void SetSourceB(I8Immediate *sourceB) { m_sourceB = sourceB; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB };
	}

	// Formatting
	
	std::string OpCode() const override { return "MOV"; }

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);

		// SourceB
		if (m_sourceB != nullptr)
		{
			code += ", ";
			code += m_sourceB->ToString();
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCode(m_sourceA,
			0x202, // Register
			0x802, // Immediate
			0xa02  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// SourceA
		code |= BinaryUtils::OperandComposite(m_sourceA);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// SourceB
		if (m_sourceB != nullptr)
		{
			code |= BinaryUtils::Format(m_sourceB->ToBinary(), 8, 0xf);
		}
		else
		{
			// Default value
			code |= BinaryUtils::Format(0xf, 8, 0xf);
		}

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::SinglePrecision; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Composite *m_sourceA = nullptr;
	I8Immediate *m_sourceB = nullptr;
};

}
}
