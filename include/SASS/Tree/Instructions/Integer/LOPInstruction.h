#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class LOPInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		NEG_I = 0x0100000000000000,
		INV   = 0x0000010000000000
	};

	enum class BooleanOperator : std::uint64_t {
		AND    = 0x0000000000000000,
		OR     = 0x0000020000000000,
		XOR    = 0x0000040000000000,
		PASS_B = 0x0000060000000000
	};

	enum class PredicateZ : std::uint64_t {
		Z    = 0x0000200000000000,
		NZ   = 0x0000300000000000
	};

	SASS_FLAGS_FRIEND()

	LOPInstruction(Register *destination, Register *sourceA, Composite *sourceB, BooleanOperator booleanOperator, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_booleanOperator(booleanOperator), m_flags(flags) {}

	LOPInstruction(Predicate *destinationP, PredicateZ predicateZ, Register *destination, Register *sourceA, Composite *sourceB, BooleanOperator booleanOperator, Flags flags = Flags::None)
		: m_destinationP(destinationP), m_predicateZ(predicateZ), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_booleanOperator(booleanOperator), m_flags(flags) {}

	// Properties

	const Predicate *GetDestinationP() const { return m_destinationP; }
	Predicate *GetDestinationP() { return m_destinationP; }
	void SetDestinationP(Predicate *destinationP) { m_destinationP = destinationP; }

	PredicateZ GetPredicateZ() const { return m_predicateZ; }
	void SetPredicateZ(PredicateZ predicateZ) { m_predicateZ = predicateZ; }

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	BooleanOperator GetBooleanOperator() const { return m_booleanOperator; }
	void SetBooleanOperator(BooleanOperator booleanOperator) { m_booleanOperator = booleanOperator; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destinationP, m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB };
	}

	// Formatting

	std::string OpCode() const override { return "LOP"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_destinationP != nullptr)
		{
			switch (m_predicateZ)
			{
				case PredicateZ::Z: code += ".Z"; break;
				case PredicateZ::NZ: code += ".NZ"; break;
			}
		}
		switch (m_booleanOperator)
		{
			case BooleanOperator::AND: code += ".AND"; break;
			case BooleanOperator::OR: code += ".OR"; break;
			case BooleanOperator::XOR: code += ".XOR"; break;
			case BooleanOperator::PASS_B: code += ".PASS_B"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// PredicateZ
		if (m_destinationP != nullptr)
		{
			code += m_destinationP->ToString();
			code += ", ";
		}
		
		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		if (m_flags & Flags::INV)
		{
			code += "~";
		}
		if (m_flags & Flags::NEG_I)
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::INV)
		{
			code += ".INV";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c40000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		auto code = BinaryUtils::OpModifierFlags(m_flags) | BinaryUtils::OpModifierFlags(m_booleanOperator);
		if (m_destinationP == nullptr)
		{
			code |= 0x0007000000000000;
		}
		else
		{
			code |= BinaryUtils::OpModifierFlags(m_predicateZ);
		}
		return code;
	}

	std::uint64_t BinaryOperands() const override
	{
		auto code = BinaryUtils::OperandRegister0(m_destination) |
		            BinaryUtils::OperandRegister8(m_sourceA) |
		            BinaryUtils::OperandComposite(m_sourceB);

		if (m_destinationP != nullptr)
		{
			code |= BinaryUtils::OperandPredicate48(m_destinationP);
		}
		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Predicate *m_destinationP = nullptr;
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;

	BooleanOperator m_booleanOperator;
	Flags m_flags = Flags::None;
	PredicateZ m_predicateZ = PredicateZ::Z;
};

SASS_FLAGS_INLINE(LOPInstruction)

}
