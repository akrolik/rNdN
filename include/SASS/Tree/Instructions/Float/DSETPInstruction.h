#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Predicate.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class DSETPInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,                    
		NEG_I = 0x0100000000000000,
		NEG_A = 0x0000080000000000,
		NEG_B = 0x0000000000000040,
		ABS_A = 0x0000000000000080,
		ABS_B = 0x0000100000000000,
		NOT_C = 0x0000040000000000
	};

	enum class ComparisonOperator : std::uint64_t {
		LT  = 0x0001000000000000,
		EQ  = 0x0002000000000000,
		LE  = 0x0003000000000000,
		GT  = 0x0004000000000000,
		NE  = 0x0005000000000000,
		GE  = 0x0006000000000000,
		NUM = 0x0007000000000000,
		NaN = 0x0008000000000000,
		LTU = 0x0009000000000000,
		EQU = 0x000a000000000000,
		LEU = 0x000b000000000000,
		GTU = 0x000c000000000000,
		NEU = 0x000d000000000000,
		GEU = 0x000e000000000000
	};

	enum class BooleanOperator : std::uint64_t {
		AND = 0x0000000000000000,
		OR  = 0x0000200000000000,
		XOR = 0x0000400000000000
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
	void SetSourceB(Predicate *sourceC) { m_sourceC = sourceC; }

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
			case ComparisonOperator::LT: code += ".LT"; break;
			case ComparisonOperator::EQ: code += ".EQ"; break;
			case ComparisonOperator::LE: code += ".LE"; break;
			case ComparisonOperator::GT: code += ".GT"; break;
			case ComparisonOperator::NE: code += ".NE"; break;
			case ComparisonOperator::GE: code += ".GE"; break;

			case ComparisonOperator::NUM: code += ".NUM"; break;
			case ComparisonOperator::NaN: code += ".NAN"; break;
			case ComparisonOperator::LTU: code += ".LTU"; break;
			case ComparisonOperator::EQU: code += ".EQU"; break;
			case ComparisonOperator::LEU: code += ".LEU"; break;
			case ComparisonOperator::GTU: code += ".GTU"; break;
			case ComparisonOperator::NEU: code += ".NEU"; break;
			case ComparisonOperator::GEU: code += ".GEU"; break;
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
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_I || m_flags & Flags::NEG_B)
		{
			code += "-";
		}
		if (m_flags & Flags::ABS_B)
		{
			code += "|";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::NEG_I && m_flags & Flags::NEG_B)
		{
			code += ".NEG";
		}
		if (m_flags & Flags::ABS_B)
		{
			code += "|";
		}
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
		return BinaryUtils::OpCodeComposite(0x5b80000000000000, m_sourceB);
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

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::Compare; }

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
