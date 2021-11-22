#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Predicate.h"
#include "SASS/Tree/Operands/I8Immediate.h"

namespace SASS {
namespace Volta {

class PLOP3Instruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0x0,
		NOT_A = (1 << 0),
		NOT_B = (1 << 1),
		NOT_C = (1 << 2)
	};

	SASS_FLAGS_FRIEND()

	PLOP3Instruction(Predicate *destinationA, Predicate *destinationB, Predicate *sourceA, Predicate *sourceB, Predicate *sourceC, I8Immediate *sourceD, I8Immediate * sourceE, Flags flags = Flags::None)
		: m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD), m_sourceE(sourceE), m_flags(flags) {}

	// Properties

	const Predicate *GetDestinationA() const { return m_destinationA; }
	Predicate *GetDestinationA() { return m_destinationA; }
	void SetDestinationA(Predicate *destinationA) { m_destinationA = destinationA; }

	const Predicate *GetDestinationB() const { return m_destinationB; }
	Predicate *GetDestinationB() { return m_destinationB; }
	void SetDestinationB(Predicate *destinationB) { m_destinationB = destinationB; }

	const Predicate *GetSourceA() const { return m_sourceA; }
	Predicate *GetSourceA() { return m_sourceA; }
	void SetSourceA(Predicate *sourceA) { m_sourceA = sourceA; }

	const Predicate *GetSourceB() const { return m_sourceB; }
	Predicate *GetSourceB() { return m_sourceB; }
	void SetSourceB(Predicate *sourceB) { m_sourceB = sourceB; }

	const Predicate *GetSourceC() const { return m_sourceC; }
	Predicate *GetSourceC() { return m_sourceC; }
	void SetSourceC(Predicate *sourceC) { m_sourceC = sourceC; }

	const I8Immediate *GetSourceD() const { return m_sourceD; }
	I8Immediate *GetSourceD() { return m_sourceD; }
	void SetSourceD(I8Immediate *sourceD) { m_sourceD = sourceD; }

	const I8Immediate *GetSourceE() const { return m_sourceE; }
	I8Immediate *GetSourceE() { return m_sourceE; }
	void SetSourceE(I8Immediate *sourceE) { m_sourceE = sourceE; }

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
	
	std::string OpCode() const override { return "PLOP3"; }

	std::string OpModifiers() const override { return ".LUT"; }

	std::string Operands() const override
	{
		std::string code;
		
		// Destination
		code += m_destinationA->ToString();
		code += ", ";
		code += m_destinationB->ToString();
		code += ", ";

		// SourceA
		if (m_flags & Flags::NOT_A)
		{
			code += "!";
		}
		code += m_sourceA->ToString();
		code += ", ";

		// SourceB
		if (m_flags & Flags::NOT_B)
		{
			code += "!";
		}
		code += m_sourceB->ToString();
		code += ", ";

		// SourceC
		if (m_flags & Flags::NOT_C)
		{
			code += "!";
		}
		code += m_sourceC->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x81c;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// SourceE
		code |= BinaryUtils::Format(m_sourceE->ToBinary(), 16, 0xff);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// DestinationA
		code |= BinaryUtils::OperandPredicate17(m_destinationA);

		// DestinationB
		code |= BinaryUtils::OperandPredicate20(m_destinationB);

		// SourceA
		code |= BinaryUtils::OperandPredicate23(m_sourceA, m_flags & Flags::NOT_A);

		// SourceB
		code |= BinaryUtils::OperandPredicate13(m_sourceB, m_flags & Flags::NOT_B);

		// SourceC
		code |= BinaryUtils::OperandPredicate4(m_sourceC, m_flags & Flags::NOT_C);

		// SourceD (split bits)
		code |= BinaryUtils::Format(m_sourceD->ToBinary(), 0, 0x7);
		code |= BinaryUtils::Format(m_sourceD->ToBinary() >> 3, 8, 0x1f);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Comparison; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Predicate *m_destinationA = nullptr;
	Predicate *m_destinationB = nullptr;
	Predicate *m_sourceA = nullptr;
	Predicate *m_sourceB = nullptr;
	Predicate *m_sourceC = nullptr;

	I8Immediate *m_sourceD = nullptr;
	I8Immediate *m_sourceE = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(PLOP3Instruction)

}
}
