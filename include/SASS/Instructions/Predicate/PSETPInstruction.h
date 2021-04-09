#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Predicate.h"

namespace SASS {

class PSETPInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0x0,
		NOT_A = 0x0000000000008000,
		NOT_B = 0x0000000100000000,
		NOT_C = 0x0000040000000000
	};

	SASS_FLAGS_FRIEND()

	enum class BooleanOperator1 : std::uint64_t {
		AND = 0x0000000000000000,
		OR  = 0x0000000001000000,
		XOR = 0x0000000002000000
	};

	enum class BooleanOperator2 : std::uint64_t {
		AND = 0x0000000000000000,
		OR  = 0x0000200000000000,
		XOR = 0x0000400000000000
	};

	PSETPInstruction(Predicate *destinationA, Predicate *destinationB, Predicate *sourceA, Predicate *sourceB, Predicate *sourceC, BooleanOperator1 booleanOperator1, BooleanOperator2 booleanOperator2, Flags flags = Flags::None)
		: PredicatedInstruction({destinationA, destinationB, sourceA, sourceB, sourceC}), m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_booleanOperator1(booleanOperator1), m_booleanOperator2(booleanOperator2), m_flags(flags) {}

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

	BooleanOperator1 GetBooleanOperator1() const { return m_booleanOperator1; }
	void SetBooleanOperator1(BooleanOperator1 booleanOperator1) { m_booleanOperator1 = booleanOperator1; }

	BooleanOperator2 GetBooleanOperator2() const { return m_booleanOperator2; }
	void SetBooleanOperator2(BooleanOperator2 booleanOperator2) { m_booleanOperator2 = booleanOperator2; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting
	
	std::string OpCode() const override { return "PSETP"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_booleanOperator1)
		{
			case BooleanOperator1::AND: code += ".AND"; break;
			case BooleanOperator1::OR: code += ".OR"; break;
			case BooleanOperator1::XOR: code += ".XOR"; break;
		}
		switch (m_booleanOperator2)
		{
			case BooleanOperator2::AND: code += ".AND"; break;
			case BooleanOperator2::OR: code += ".OR"; break;
			case BooleanOperator2::XOR: code += ".XOR"; break;
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
		return 0x5090000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_booleanOperator1) |
		       BinaryUtils::OpModifierFlags(m_booleanOperator2) |
		       BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandPredicate3(m_destinationA) |
		       BinaryUtils::OperandPredicate0(m_destinationB) |
		       BinaryUtils::OperandPredicate12(m_sourceA) |
		       BinaryUtils::OperandPredicate29(m_sourceB) |
		       BinaryUtils::OperandPredicate39(m_sourceC);
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::Compare; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }

private:
	Predicate *m_destinationA = nullptr;
	Predicate *m_destinationB = nullptr;
	Predicate *m_sourceA = nullptr;
	Predicate *m_sourceB = nullptr;
	Predicate *m_sourceC = nullptr;

	BooleanOperator1 m_booleanOperator1;
	BooleanOperator2 m_booleanOperator2;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(PSETPInstruction)

}
