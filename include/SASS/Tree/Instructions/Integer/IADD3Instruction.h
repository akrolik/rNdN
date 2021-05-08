#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/CarryFlag.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class IADD3Instruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		CC    = 0x0000800000000000,
		X     = 0x0001000000000000,
		RS    =	0x0000002000000000,
		LS    = 0x0000004000000000,

		//TODO: Bit formats
		H0_A  = 0x0000000000000000,
		H1_A  = 0x0000000000000000,
		H0_B  = 0x0000000000000000,
		H1_B  = 0x0000000000000000,
		H0_C  = 0x0000000000000000,
		H1_C  = 0x0000000000000000,

		NEG_I = 0x0100000000000000,
		NEG_A = 0x0008000000000000,
		NEG_B = 0x0004000000000000,
		NEG_C = 0x0002000000000000
	};

	SASS_FLAGS_FRIEND()

	IADD3Instruction(Register *destination, Register *sourceA, Composite *sourceB, Register *sourceC, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	const Register *GetSourceC() const { return m_sourceC; }
	Register *GetSourceC() { return m_sourceC; }
	void SetSourceC(Register *sourceC) { m_sourceC = sourceC; }

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
			return { m_sourceA, m_sourceB, m_sourceC, SASS::CC };
		}
		return { m_sourceA, m_sourceB, m_sourceC };
	}

	// Formatting

	std::string OpCode() const override { return "IADD3"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::X)
		{
			code += ".X";
		}
		if (m_flags & Flags::LS)
		{
			code += ".LS";
		}
		if (m_flags & Flags::RS)
		{
			code += ".RS";
		}
		return code;
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
		code += ", ";

		// SourceA
		if (m_flags & Flags::NEG_A)
		{
			code += "-";
		}
		code += m_sourceA->ToString();
		if (m_flags & Flags::H0_A)
		{
			code += ".H0";
		}
		if (m_flags & Flags::H1_A)
		{
			code += ".H1";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_I || m_flags & Flags::NEG_B)
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::NEG_I && m_flags & Flags::NEG_B)
		{
			code += ".NEG";
		}
		if (m_flags & Flags::H0_B)
		{
			code += ".H0";
		}
		if (m_flags & Flags::H1_B)
		{
			code += ".H1";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);
		code += ", ";

		// SourceC
		if (m_flags & Flags::NEG_C)
		{
			code += "-";
		}
		code += m_sourceC->ToString();
		if (m_flags & Flags::H0_C)
		{
			code += ".H0";
		}
		if (m_flags & Flags::H1_C)
		{
			code += ".H1";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5cc0000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB) |
		       BinaryUtils::OperandRegister39(m_sourceC);
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::Core; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Register *m_sourceC = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(IADD3Instruction)

}
