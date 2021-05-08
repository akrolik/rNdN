#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/I32Immediate.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class LOP32IInstruction : public PredicatedInstruction
{
public:
	enum class BooleanOperator : std::uint64_t {
		AND    = 0x0000000000000000,
		OR     = 0x0020000000000000,
		XOR    = 0x0040000000000000,
	};

	LOP32IInstruction(Register *destination, Register *sourceA, I32Immediate *sourceB, BooleanOperator booleanOperator)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_booleanOperator(booleanOperator) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const I32Immediate *GetSourceB() const { return m_sourceB; }
	I32Immediate *GetSourceB() { return m_sourceB; }
	void SetSourceB(I32Immediate *sourceB) { m_sourceB = sourceB; }

	BooleanOperator GetBooleanOperator() const { return m_booleanOperator; }
	void SetBooleanOperator(BooleanOperator booleanOperator) { m_booleanOperator = booleanOperator; }

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
	
	std::string OpCode() const override { return "LOP32I"; }

	std::string OpModifiers() const override
	{
		switch (m_booleanOperator)
		{
			case BooleanOperator::AND: return ".AND";
			case BooleanOperator::OR: return ".OR";
			case BooleanOperator::XOR: return ".XOR";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// Source A
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// Source B
		code += m_sourceB->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x0400000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_booleanOperator);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandLiteral20W32(m_sourceB);
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::Core; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	I32Immediate *m_sourceB = nullptr;

	BooleanOperator m_booleanOperator;
};

}
