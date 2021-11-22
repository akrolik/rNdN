#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Maxwell {

class ICMPInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,                    
		U32   = 0x0001000000000000
	};

	enum class ComparisonOperator : std::uint64_t {
		LT = 0x0002000000000000,
		EQ = 0x0004000000000000,
		LE = 0x0006000000000000,
		GT = 0x0008000000000000,
		NE = 0x000a000000000000,
		GE = 0x000c000000000000
	};

	SASS_FLAGS_FRIEND()

	ICMPInstruction(Register *destination, Register *sourceA, Composite *sourceB, Register *sourceC, ComparisonOperator comparisonOperator, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_comparisonOperator(comparisonOperator), m_flags(flags) {}

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
	void SetSourceB(Register *sourceC) { m_sourceC = sourceC; }

	ComparisonOperator GetComparisonOperator() const { return m_comparisonOperator; }
	void SetComparisonOperator(ComparisonOperator comparisonOperator) { m_comparisonOperator = comparisonOperator; }

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
	
	std::string OpCode() const override { return "ICMP"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_comparisonOperator)
		{
			case ComparisonOperator::LT: code += ".LT"; break;
			case ComparisonOperator::EQ: code += ".EQ"; break;
			case ComparisonOperator::LE: code += ".LE"; break;
			case ComparisonOperator::GT: code += ".GT"; break;
			case ComparisonOperator::NE: code += ".NE"; break;
			case ComparisonOperator::GE: code += ".GE"; break;
		}
		if (m_flags & Flags::U32)
		{
			code += ".U32";
		}
		return code;
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
		code += ", ";

		// SourceC
		code += m_sourceC->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5b41000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		auto code = BinaryUtils::OpModifierFlags(m_comparisonOperator) |
			    BinaryUtils::OpModifierFlags(m_flags) ^ Flags::U32; // Flipped bit pattern

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
		       BinaryUtils::OperandComposite(m_sourceB) |
		       BinaryUtils::OperandRegister39(m_sourceC);
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Comparison; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Register *m_sourceC = nullptr;

	ComparisonOperator m_comparisonOperator;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(ICMPInstruction)

}
}
