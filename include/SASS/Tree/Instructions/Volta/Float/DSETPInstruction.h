#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Predicate.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class DSETPInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,                    
		NEG_A = (1 << 0),
		NEG_B = (1 << 1),
		ABS_A = (1 << 2),
		ABS_B = (1 << 3),
		NOT_C = (1 << 4)
	};

	enum class ComparisonOperator : std::uint64_t {
		MIN = 0x0,
		LT  = 0x1,
		EQ  = 0x2,
		LE  = 0x3,
		GT  = 0x4,
		NE  = 0x5,
		GE  = 0x6,
		NUM = 0x7,
		NaN = 0x8,
		LTU = 0x9,
		EQU = 0xa,
		LEU = 0xb,
		GTU = 0xc,
		NEU = 0xd,
		GEU = 0xe,
		MAX = 0xf
	};

	enum class BooleanOperator : std::uint64_t {
		AND = 0x0,
		OR  = 0x1,
		XOR = 0x2
	};

	SASS_FLAGS_FRIEND()

	DSETPInstruction(Predicate *destinationA, Predicate *destinationB, Register *sourceA, Composite *sourceB, Predicate *sourceC, ComparisonOperator comparisonOperator, BooleanOperator booleanOperator, Flags flags = Flags::None)
		: m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_comparisonOperator(comparisonOperator), m_booleanOperator(booleanOperator), m_flags(flags) {}

	// Properties

	const Predicate *GetDestinationA() const { return m_destinationA; }
	Predicate *GetDestinationA() { return m_destinationA; }
	void SetDestinationA(Predicate *destinationA) { m_destinationA = destinationA; }

	const Predicate *GetDestinationB() const { return m_destinationB; }
	Predicate *GetDestinationB() { return m_destinationB; }
	void SetDestinationB(Predicate *destinationB) { m_destinationB = destinationB; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	const Predicate *GetSourceC() const { return m_sourceC; }
	Predicate *GetSourceC() { return m_sourceC; }
	void SetSourceC(Predicate *sourceC) { m_sourceC = sourceC; }

	ComparisonOperator GetComparisonOperator() const { return m_comparisonOperator; }
	void SetComparisonOperator(ComparisonOperator comparisonOperator) { m_comparisonOperator = comparisonOperator; }

	BooleanOperator GetBooleanOperator() const { return m_booleanOperator; }
	void SetBooleanOperator(BooleanOperator booleanOperator) { m_booleanOperator = booleanOperator; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destinationA, m_destinationB };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB, m_sourceC };
	}

	// Formatting
	
	std::string OpCode() const override { return "DSETP"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_comparisonOperator)
		{
			case ComparisonOperator::MAX: code += ".MAX"; break;
			case ComparisonOperator::LT:  code += ".LT"; break;
			case ComparisonOperator::EQ:  code += ".EQ"; break;
			case ComparisonOperator::LE:  code += ".LE"; break;
			case ComparisonOperator::GT:  code += ".GT"; break;
			case ComparisonOperator::NE:  code += ".NE"; break;
			case ComparisonOperator::GE:  code += ".GE"; break;
			case ComparisonOperator::NUM: code += ".NUM"; break;
			case ComparisonOperator::NaN: code += ".NAN"; break;
			case ComparisonOperator::LTU: code += ".LTU"; break;
			case ComparisonOperator::EQU: code += ".EQU"; break;
			case ComparisonOperator::LEU: code += ".LEU"; break;
			case ComparisonOperator::GTU: code += ".GTU"; break;
			case ComparisonOperator::NEU: code += ".NEU"; break;
			case ComparisonOperator::GEU: code += ".GEU"; break;
			case ComparisonOperator::MIN: code += ".MIN"; break;
		}
		switch (m_booleanOperator)
		{
			case BooleanOperator::AND: code += ".AND"; break;
			case BooleanOperator::OR: code += ".OR"; break;
			case BooleanOperator::XOR: code += ".XOR"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;
		
		// DestinationA/B
		code += m_destinationA->ToString();
		code += ", ";
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
			0x22a, // Register
			0x42a, // Immediate
			0x62a  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

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

		// Boolean
		code |= BinaryUtils::Format(m_booleanOperator, 10, 0x3);

		// Comparison
		code |= BinaryUtils::Format(m_comparisonOperator, 12, 0xf);

		// DestinationA
		code |= BinaryUtils::OperandPredicate17(m_destinationA);

		// DestinationB
		code |= BinaryUtils::OperandPredicate20(m_destinationB);

		// SourceC
		code |= BinaryUtils::OperandPredicate23(m_sourceC, m_flags & Flags::NOT_C); 

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::DoublePrecision; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Predicate *m_destinationA = nullptr;
	Predicate *m_destinationB = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Predicate *m_sourceC = nullptr;

	ComparisonOperator m_comparisonOperator;
	BooleanOperator m_booleanOperator;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(DSETPInstruction)

}
}
