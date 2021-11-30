#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

class IADD3Instruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		X     = (1 << 0),
		NEG_A = (1 << 1),
		NEG_B = (1 << 2),
		INV_B = (1 << 3),
		NEG_C = (1 << 4),
		NOT_D = (1 << 5),
		NOT_E = (1 << 6)
	};

	SASS_FLAGS_FRIEND()

	// Full
	IADD3Instruction(Register *destinationA, Predicate *destinationB, Predicate *destinationC, Register *sourceA, Composite *sourceB, Register *sourceC, Predicate *sourceD, Predicate *sourceE, Flags flags = Flags::None)
		: m_destinationA(destinationA), m_destinationB(destinationB), m_destinationC(destinationC), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD), m_sourceE(sourceE), m_flags(flags) {}

	// Add
	IADD3Instruction(Register *destinationA, Register *sourceA, Composite *sourceB, Register *sourceC, Flags flags = Flags::None)
		: IADD3Instruction(destinationA, nullptr, nullptr, sourceA, sourceB, sourceC, nullptr, nullptr, flags) {}

	// Carry-out
	IADD3Instruction(Register *destinationA, Predicate *destinationB, Register *sourceA, Composite *sourceB, Register *sourceC, Flags flags = Flags::None)
		: IADD3Instruction(destinationA, destinationB, nullptr, sourceA, sourceB, sourceC, nullptr, nullptr, flags) {}

	// Carry-in
	IADD3Instruction(Register *destinationA, Register *sourceA, Composite *sourceB, Register *sourceC, Predicate *sourceD, Predicate *sourceE, Flags flags = Flags::None)
		: IADD3Instruction(destinationA, nullptr, nullptr, sourceA, sourceB, sourceC, sourceD, sourceE, flags | Flags::X) {}

	// Properties

	const Register *GetDestinationA() const { return m_destinationA; }
	Register *GetDestinationA() { return m_destinationA; }
	void SetDestinationA(Register *destinationA) { m_destinationA = destinationA; }

	const Predicate *GetDestinationB() const { return m_destinationB; }
	Predicate *GetDestinationB() { return m_destinationB; }
	void SetDestinationB(Predicate *destinationB) { m_destinationB = destinationB; }

	const Predicate *GetDestinationC() const { return m_destinationC; }
	Predicate *GetDestinationC() { return m_destinationC; }
	void SetDestinationC(Predicate *destinationC) { m_destinationC = destinationC; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	const Register *GetSourceC() const { return m_sourceC; }
	Register *GetSourceC() { return m_sourceC; }
	void SetSourceC(Register *sourceC) { m_sourceC = sourceC; }

	const Predicate *GetSourceD() const { return m_sourceD; }
	Predicate *GetSourceD() { return m_sourceD; }
	void SetSourceD(Predicate *sourceD) { m_sourceD = sourceD; }

	const Predicate *GetSourceE() const { return m_sourceE; }
	Predicate *GetSourceE() { return m_sourceE; }
	void SetSourceC(Predicate *sourceE) { m_sourceE = sourceE; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destinationA, m_destinationB, m_destinationC };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB, m_sourceC, m_sourceD, m_sourceE };
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
		return code;
	}

	std::string Operands() const override
	{             
		std::string code;

		// DestinationA
		code += m_destinationA->ToString();
		code += ", ";

		// DestinationB
		if (m_destinationB != nullptr)
		{
			code += m_destinationB->ToString();
			code += ", ";
		}

		// DestinationC
		if (m_destinationC != nullptr)
		{
			code += m_destinationC->ToString();
			code += ", ";
		}

		// SourceA
		if (m_flags & Flags::NEG_A)
		{
			code += "-";
		}
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_B)
		{
			code += "-";
		}

		if (m_flags & Flags::X && m_flags & Flags::INV_B)
		{
			code += "~";
		}
		code += m_sourceB->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);
		code += ", ";

		// SourceC
		if (m_flags & Flags::NEG_C)
		{
			code += "-";
		}
		code += m_sourceC->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);

		// SourceD
		if (m_sourceD != nullptr)
		{
			code += ", ";
			if (m_flags & Flags::NOT_D)
			{
				code += "!";
			}
			code += m_sourceD->ToString();
		}

		// SourceE
		if (m_sourceE != nullptr)
		{
			code += ", ";
			if (m_flags & Flags::NOT_E)
			{
				code += "!";
			}
			code += m_sourceE->ToString();
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCode(m_sourceB,
			0x210, // Register
			0x810, // Immediate
			0xa10  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// DestinationA
		code |= BinaryUtils::OperandRegister16(m_destinationA);

		// SourceA
		code |= BinaryUtils::OperandRegister24(m_sourceA);

		// SourceB
		code |= BinaryUtils::OperandComposite(m_sourceB, m_flags & Flags::NEG_B);
		if (m_sourceB->GetKind() != Operand::Kind::Immediate)
		{
			code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_B || m_flags & Flags::INV_B, 63);
		}

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Flags
		code |= BinaryUtils::FlagBit(m_flags & Flags::X, 10);
		code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_A, 8);
		code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_C, 11);

		// SourceC
		code |= BinaryUtils::OperandRegister0(m_sourceC);

		// SourceD
		code |= BinaryUtils::OperandPredicate23(m_sourceD, m_flags & Flags::NOT_D);

		// SourceE
		code |= BinaryUtils::OperandPredicate13(m_sourceE, m_flags & Flags::NOT_E);

		// DestinationB
		code |= BinaryUtils::OperandPredicate17(m_destinationB);

		// DestinationC
		code |= BinaryUtils::OperandPredicate20(m_destinationC);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destinationA = nullptr;
	Predicate *m_destinationB = nullptr;
	Predicate *m_destinationC = nullptr;

	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Register *m_sourceC = nullptr;
	Predicate *m_sourceD = nullptr;
	Predicate *m_sourceE = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(IADD3Instruction)

}
}
