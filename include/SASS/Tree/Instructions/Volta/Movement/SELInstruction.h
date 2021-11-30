#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

class SELInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,                    
		NOT_C = (1 << 0)
	};

	SASS_FLAGS_FRIEND()

	SELInstruction(Register *destination, Register *sourceA, Composite *sourceB, Predicate *sourceC, Flags flags = Flags::None)
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

	const Predicate *GetSourceC() const { return m_sourceC; }
	Predicate *GetSourceC() { return m_sourceC; }
	void SetSourceC(Predicate *sourceC) { m_sourceC = sourceC; }

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
	
	std::string OpCode() const override { return "SEL"; }

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
		return BinaryUtils::OpCode(m_sourceB,
			0x207, // Register
			0x807, // Immediate
			0xa07  // Constant
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
		code |= BinaryUtils::OperandComposite(m_sourceB);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// SourceC
		code |= BinaryUtils::OperandPredicate23(m_sourceC, m_flags & Flags::NOT_C);

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
	Predicate *m_sourceC = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(SELInstruction)

}
}
