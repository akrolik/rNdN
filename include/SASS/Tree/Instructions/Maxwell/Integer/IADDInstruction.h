#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/CarryFlag.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Maxwell {

class IADDInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		SAT   = 0x0004000000000000,
		X     = 0x0000080000000000,
		CC    = 0x0000800000000000,
		NEG_A = 0x0002000000000000,
		NEG_B = 0x0001000000000000
	};

	SASS_FLAGS_FRIEND()

	IADDInstruction(Register *destination, Register *sourceA, Composite *sourceB, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_flags(flags) {}

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

	std::string OpCode() const override { return "IADD"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::SAT)
		{
			code += ".SAT";
		}
		if (m_flags & Flags::X)
		{
			code += ".X";
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
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		if ((m_flags & Flags::NEG_B) && !m_sourceB->GetOpModifierNegate())
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		if ((m_flags & Flags::NEG_B) && m_sourceB->GetOpModifierNegate())
		{
			code += ".NEG";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c10000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		auto code = BinaryUtils::OpModifierFlags(m_flags);
		if (m_sourceB->GetOpModifierNegate())
		{
			code |= 0x0100000000000000;
		}
		return code;
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB);
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(IADDInstruction)

}
}
