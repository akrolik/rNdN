#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class DFMAInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		NEG_A = (1 << 0),
		NEG_B = (1 << 1),
		NEG_C = (1 << 2),
		ABS_A = (1 << 3),
		ABS_B = (1 << 4),
		ABS_C = (1 << 5)
	};

	SASS_FLAGS_FRIEND()

	enum class Round : std::uint64_t {
		RN = 0x0,
		RM = 0x1,
		RP = 0x2,
		RZ = 0x3
	};

	DFMAInstruction(Register *destination, Register *sourceA, Composite *sourceB, Composite *sourceC, Round round = Round::RN, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_round(round), m_flags(flags) {}

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

	const Composite *GetSourceC() const { return m_sourceC; }
	Composite *GetSourceC() { return m_sourceC; }
	void SetSourceC(Composite *sourceC) { m_sourceC = sourceC; }

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
		return { m_sourceA, m_sourceB, m_sourceC };
	}

	// Formatting

	std::string OpCode() const override { return "DFMA"; }

	std::string OpModifiers() const override
	{
		switch (m_round)
		{
			case Round::RN: return ".RN";
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
		code += ", ";

		// SourceC
		if (m_flags & Flags::NEG_C)
		{
			code += "-";
		}
		if (m_flags & Flags::ABS_C)
		{
			code += "|";
		}
		code += m_sourceC->ToString();
		if (m_flags & Flags::ABS_C)
		{
			code += "|";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		if (m_sourceB->GetKind() == Operand::Kind::Immediate)
		{
			return 0x82b;
		}
		else if (m_sourceB->GetKind() == Operand::Kind::Constant)
		{
			return 0xa2b;
		}
		else if (m_sourceC->GetKind() == Operand::Kind::Immediate)
		{
			return 0x42b;
		}
		else if (m_sourceC->GetKind() == Operand::Kind::Constant)
		{
			return 0x62b;
		}
		return 0x22b;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// SourceA
		code |= BinaryUtils::OperandRegister24(m_sourceA);

		// SourceB/SourceC (immediate/constant)
		if (m_sourceC->GetKind() == Operand::Kind::Register)
		{
			code |= BinaryUtils::OperandComposite(m_sourceB);

			// Flags (SourceB constant)
			if (m_sourceB->GetKind() != Operand::Kind::Immediate)
			{
				code |= BinaryUtils::FlagBit(m_flags & Flags::ABS_B, 62);
				code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_B, 63);
			}
		}
		else
		{
			code |= BinaryUtils::OperandComposite(m_sourceC);

			// Flags (SourceC)
			if (m_sourceC->GetKind() != Operand::Kind::Immediate)
			{
				code |= BinaryUtils::FlagBit(m_flags & Flags::ABS_C, 62);
				code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_C, 63);
			}
		}

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// SourceC (register)/SourceB (if SourceC not register)
		if (m_sourceC->GetKind() == Operand::Kind::Register)
		{
			auto registerC = static_cast<Register *>(m_sourceC);
			code |= BinaryUtils::OperandRegister0(registerC);

			// Flags
			code |= BinaryUtils::FlagBit(m_flags & Flags::ABS_C, 10);
			code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_C, 11);
		}
		else
		{
			auto registerB = static_cast<Register *>(m_sourceB);
			code |= BinaryUtils::OperandRegister0(registerB);

			// Flags
			code |= BinaryUtils::FlagBit(m_flags & Flags::ABS_B, 10);
			code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_B, 11);
		}

		// Flags (SourceA)
		code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_A, 8);
		code |= BinaryUtils::FlagBit(m_flags & Flags::ABS_A, 9);

		// Rounding modifier
		code |= BinaryUtils::Format(m_round, 14, 0x3);

		return code;
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
	Composite *m_sourceC = nullptr;

	Round m_round = Round::RN;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(DFMAInstruction)

}
}
