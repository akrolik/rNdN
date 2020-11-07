#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Predicate.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class ISETPInstruction : public Instruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,                    
		X     = 0x0000080000000000,
		U32   = 0x0001000000000000,
		NEG_I = 0x0100000000000000,
		NOT_C = 0x0000040000000000
	};

	enum class ComparisonOperator : std::uint64_t {
		LT = 0x0002000000000000,
		EQ = 0x0004000000000000,
		LE = 0x0006000000000000,
		GT = 0x0008000000000000,
		NE = 0x000a000000000000,
		GE = 0x000c000000000000
	};

	enum class BooleanOperator : std::uint64_t {
		AND = 0x0000000000000000,
		OR  = 0x0000200000000000,
		XOR = 0x0000400000000000
	};

	SASS_FLAGS_FRIEND()

	ISETPInstruction(const Predicate *destinationA, const Predicate *destinationB, const Register *sourceA, const Composite *sourceB, const Predicate *sourceC, ComparisonOperator comparisonOperator, BooleanOperator booleanOperator, Flags flags = Flags::None)
		: Instruction({destinationA, destinationB, sourceA, sourceB, sourceC}), m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_comparisonOperator(comparisonOperator), m_booleanOperator(booleanOperator), m_flags(flags) {}

	// Properties

	const Predicate *GetDestinationA() const { return m_destinationA; }
	void SetDestinationA(const Predicate *destinationA) { m_destinationA = destinationA; }

	const Predicate *GetDestinationB() const { return m_destinationB; }
	void SetDestinationB(const Predicate *destinationB) { m_destinationB = destinationB; }

	const Register *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const Composite *sourceB) { m_sourceB = sourceB; }

	const Predicate *GetSourceC() const { return m_sourceC; }
	void SetSourceB(const Predicate *sourceC) { m_sourceC = sourceC; }

	ComparisonOperator GetComparisonOperator() const { return m_comparisonOperator; }
	void SetComparisonOperator(ComparisonOperator comparisonOperator) { m_comparisonOperator = comparisonOperator; }

	BooleanOperator GetBooleanOperator() const { return m_booleanOperator; }
	void SetBooleanOperator(BooleanOperator booleanOperator) { m_booleanOperator = booleanOperator; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting
	
	std::string OpCode() const override { return "ISETP"; }

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
		if (m_flags & Flags::X)
		{
			code += ".X";
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
		
		// Destination
		code += m_destinationA->ToString();
		code += ", ";
		code += m_destinationB->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_I)
		{
			code += "-";
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
		return BinaryUtils::OpCodeComposite(0x5b61000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_comparisonOperator) |
		       BinaryUtils::OpModifierFlags(m_booleanOperator) |
		       BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandPredicate3(m_destinationA) |
		       BinaryUtils::OperandPredicate0(m_destinationB) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB) |
		       BinaryUtils::OperandPredicate39(m_sourceC);
	}

private:
	const Predicate *m_destinationA = nullptr;
	const Predicate *m_destinationB = nullptr;
	const Register *m_sourceA = nullptr;
	const Composite *m_sourceB = nullptr;
	const Predicate *m_sourceC = nullptr;

	ComparisonOperator m_comparisonOperator;
	BooleanOperator m_booleanOperator;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(ISETPInstruction)

}
