#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Maxwell {

class SHLInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		W     = 0x0000008000000000
	};

	SASS_FLAGS_FRIEND()

	SHLInstruction(Register *destination, Register *sourceA, Composite *sourceB, Flags flags = Flags::None)
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
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB };
	}

	// Formatting

	std::string OpCode() const override { return "SHL"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::W)
		{
			return ".W";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		code += m_sourceB->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c48000000000000, m_sourceB);
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

	InstructionClass GetInstructionClass() const override { return InstructionClass::Shift; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(SHLInstruction)

}
}
