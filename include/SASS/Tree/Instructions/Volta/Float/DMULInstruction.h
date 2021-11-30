#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class DMULInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		NEG_A = (1 << 0),
		NEG_B = (1 << 1),
		ABS_A = (1 << 2),
		ABS_B = (1 << 3)
	};

	enum class Round : std::uint64_t {
		RN = 0x0,
		RM = 0x1,
		RP = 0x2,
		RZ = 0x3
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
		if (m_flags & Flags::NEG_A)
		{
			code += "-";
		}
		if (m_flags & Flags::ABS_A)
		{
			code += "|";
		}
		code += m_sourceA->ToString();
		if (m_flags & Flags::ABS_A)
		{
			code += "|";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_B)
		{
			code += "-";
		}
		if (m_flags & Flags::ABS_B)
		{
			code += "|";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::ABS_B)
		{
			code += "|";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCode(m_sourceB,
			0x228, // Register
			0x828, // Immediate
			0xa28  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// SourceA
		code |= BinaryUtils::OperandRegister24(m_sourceA);

		// SourceB
		code |= BinaryUtils::OperandComposite(m_sourceB, m_flags & Flags::NEG_B, m_flags & Flags::ABS_B);

		// Flags (SourceB register/constant)
		if (m_sourceB->GetKind() != Operand::Kind::Immediate)
		{
			code |= BinaryUtils::FlagBit(m_flags & Flags::ABS_B, 62);
			code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_B, 63);
		}

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Flags (SourceA)
		code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_A, 8);
		code |= BinaryUtils::FlagBit(m_flags & Flags::ABS_A, 9);

		// Rounding modifier
		code |= BinaryUtils::Format(m_round, 14, 0x3);

		return code;
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
}
