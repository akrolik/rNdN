#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/CarryFlag.h"
#include "SASS/Tree/Operands/I32Immediate.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class IADD32IInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		X     = 0x0020000000000000,
		CC    = 0x0010000000000000,
		NEG_A = 0x0100000000000000
	};

	SASS_FLAGS_FRIEND()

	IADD32IInstruction(Register *destination, Register *sourceA, I32Immediate *sourceB, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_flags(flags) {}

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
	
	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		if (m_flags & Flags::CC)
		{
			return { m_destination, SASS::CC };
		}
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		if (m_flags & Flags::X)
		{
			return { m_sourceA, m_sourceB, SASS::CC };
		}
		return { m_sourceA, m_sourceB };
	}

	// Formatting

	std::string OpCode() const override { return "IADD32I"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::X)
		{
			return ".X";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		if (m_flags & Flags::CC)
		{
			code += ".CC";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		if (m_flags & Flags::NEG_A)
		{
			code += "-";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);
		code += ", ";

		// SourceB
		code += m_sourceB->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x1c00000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandLiteral20W32(m_sourceB);
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	I32Immediate *m_sourceB = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(IADD32IInstruction)

}
