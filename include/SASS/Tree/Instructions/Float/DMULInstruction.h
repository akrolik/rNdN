#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class DMULInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		NEG_I = 0x0100000000000000,
		NEG_B = 0x0001000000000000
	};

	enum class Round : std::uint64_t {
		RN = 0x0000000000000000,
		RM = 0x0000008000000000,
		RP = 0x0000010000000000,
		RZ = 0x0000018000000000
	};

	SASS_FLAGS_FRIEND()

	DMULInstruction(Register *destination, Register *sourceA, Composite *sourceB, Round round = Round::RN, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_round(round), m_flags(flags) {}

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

	Round GetRound() const { return m_round; }
	void SetRound(Round round) { m_round = round; }

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
	
	std::string OpCode() const override { return "DMUL"; }

	std::string OpModifiers() const override
	{
		switch (m_round)
		{
			case Round::RN: return "";
			case Round::RM: return ".RM";
			case Round::RP: return ".RP";
			case Round::RZ: return ".RZ";
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
		if (m_flags & Flags::NEG_I || m_flags & Flags::NEG_B)
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::NEG_I && m_flags & Flags::NEG_B)
		{
			code += ".NEG";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);
		return code;                               
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c80000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_round) | BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB);
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::DoublePrecision; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;

	Round m_round = Round::RN;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(DMULInstruction)

}
