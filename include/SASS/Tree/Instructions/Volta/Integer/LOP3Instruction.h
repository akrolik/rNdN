#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

class LOP3Instruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None   = 0,
		NOT_E  = (1 << 0)
	};

	SASS_FLAGS_FRIEND()

	LOP3Instruction(Predicate *destinationA, Register *destinationB, Register *sourceA, Composite *sourceB, Register *sourceC, I8Immediate *sourceD, Predicate *sourceE, Flags flags = Flags::None)
		: m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD), m_sourceE(sourceE), m_flags(flags) {}

	LOP3Instruction(Register *destinationB, Register *sourceA, Composite *sourceB, Register *sourceC, I8Immediate *sourceD, Predicate *sourceE, Flags flags = Flags::None)
		: LOP3Instruction(nullptr, destinationB, sourceA, sourceB, sourceC, sourceD, sourceE, flags) {}

	// Properties

	const Predicate *GetDestinationA() const { return m_destinationA; }
	Predicate *GetDestinationA() { return m_destinationA; }
	void SetDestinationA(Predicate *destinationA) { m_destinationA = destinationA; }

	const Register *GetDestinationB() const { return m_destinationB; }
	Register *GetDestinationB() { return m_destinationB; }
	void SetDestinationB(Register *destinationB) { m_destinationB = destinationB; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	const Register *GetSourceC() const { return m_sourceC; }
	Register *GetSourceC() { return m_sourceC; }
	void SetSourceC(Register *sourceC) { m_sourceC = sourceC; }

	const I8Immediate *GetSourceD() const { return m_sourceD; }
	I8Immediate *GetSourceD() { return m_sourceD; }
	void SetSourceD(I8Immediate *sourceD) { m_sourceD = sourceD; }

	const Predicate *GetSourceE() const { return m_sourceE; }
	Predicate *GetSourceE() { return m_sourceE; }
	void SetSourceE(Predicate *sourceE) { m_sourceE = sourceE; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destinationA, m_destinationB };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB, m_sourceC, m_sourceD, m_sourceE };
	}

	// Formatting

	std::string OpCode() const override { return "LOP3"; }

	std::string OpModifiers() const override { return ".LUT"; }

	std::string Operands() const override
	{
		std::string code;

		// DestinationA
		if (m_destinationA != nullptr)
		{
			code += m_destinationA->ToString();
			code += ", ";
		}
		
		// DestinationB
		code += m_destinationB->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		code += m_sourceB->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);
		code += ", ";

		// SourceC
		code += m_sourceC->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);
		code += ", ";

		// SourceD
		code += m_sourceD->ToString();
		code += ", ";

		// SourceE
		if (m_flags & Flags::NOT_E)
		{
			code += "!";
		}
		code += m_sourceE->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCode(m_sourceB,
			0x212, // Register
			0x812, // Immediate
			0xa12  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// DestinationB
		code |= BinaryUtils::OperandRegister16(m_destinationB);

		// SourceA
		code |= BinaryUtils::OperandRegister24(m_sourceA);

		// SourceB
		code |= BinaryUtils::OperandComposite(m_sourceB);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// SourceC
		code |= BinaryUtils::OperandRegister0(m_sourceC);

		// SourceD
		code |= BinaryUtils::Format(m_sourceD->ToBinary(), 8, 0xff);

		// SourceE
		code |= BinaryUtils::OperandPredicate23(m_sourceE, m_flags & Flags::NOT_E);

		// DestinationA
		code |= BinaryUtils::OperandPredicate17(m_destinationA);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Predicate *m_destinationA = nullptr;
	Register *m_destinationB = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Register *m_sourceC = nullptr;
	I8Immediate *m_sourceD = nullptr;
	Predicate *m_sourceE = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(LOP3Instruction)

}
}
