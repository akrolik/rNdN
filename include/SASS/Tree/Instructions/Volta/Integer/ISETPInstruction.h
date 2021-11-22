#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Predicate.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class ISETPInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,                    
		EX    = (1 << 0),
		U32   = (1 << 1),
		NOT_C = (1 << 2),
		NOT_D = (1 << 3)
	};

	enum class ComparisonOperator : std::uint64_t {
		F  = 0x0,
		LT = 0x1,
		EQ = 0x2,
		LE = 0x3,
		GT = 0x4,
		NE = 0x5,
		GE = 0x6,
		T  = 0x7
	};

	enum class BooleanOperator : std::uint64_t {
		AND = 0x0,
		OR  = 0x1,
		XOR = 0x2
	};

	SASS_FLAGS_FRIEND()

	ISETPInstruction(Predicate *destinationA, Predicate *destinationB, Register *sourceA, Composite *sourceB, Predicate *sourceC, Predicate *sourceD, ComparisonOperator comparisonOperator, BooleanOperator booleanOperator, Flags flags = Flags::None)
		: m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD), m_comparisonOperator(comparisonOperator), m_booleanOperator(booleanOperator), m_flags(flags) {}

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

	const Predicate *GetSourceD() const { return m_sourceD; }
	Predicate *GetSourceD() { return m_sourceD; }
	void SetSourceD(Predicate *sourceD) { m_sourceD = sourceD; }

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
		return { m_sourceA, m_sourceB, m_sourceC, m_sourceD };
	}

	// Formatting
	
	std::string OpCode() const override { return "ISETP"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_comparisonOperator)
		{
			case ComparisonOperator::F:  code += ".F"; break;
			case ComparisonOperator::LT: code += ".LT"; break;
			case ComparisonOperator::EQ: code += ".EQ"; break;
			case ComparisonOperator::LE: code += ".LE"; break;
			case ComparisonOperator::GT: code += ".GT"; break;
			case ComparisonOperator::NE: code += ".NE"; break;
			case ComparisonOperator::GE: code += ".GE"; break;
			case ComparisonOperator::T:  code += ".T"; break;
		}
		if (m_flags & Flags::U32)
		{
			code += ".U32";
		}
		switch (m_booleanOperator)
		{
			case BooleanOperator::AND: code += ".AND"; break;
			case BooleanOperator::OR: code += ".OR"; break;
			case BooleanOperator::XOR: code += ".XOR"; break;
		}
		if (m_flags & Flags::EX)
		{
			code += ".EX";
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

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCode(m_sourceB,
			0x20c, // Register
			0x80c, // Immediate
			0xa0c  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// SourceA
		code |= BinaryUtils::OperandRegister24(m_sourceA);

		// SourceB
		code |= BinaryUtils::OperandComposite(m_sourceB);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// SourceD
		code |= BinaryUtils::OperandPredicate4(m_sourceD, m_flags & Flags::NOT_D);

		// Flags
		code |= BinaryUtils::Format(m_flags, 8, 0x3);

		// Boolean
		code |= BinaryUtils::Format(m_booleanOperator, 10, 0x3);

		// Comparison
		code |= BinaryUtils::Format(m_comparisonOperator, 12, 0x7);

		// DestinationA
		code |= BinaryUtils::OperandPredicate17(m_destinationA);

		// DestinationB
		code |= BinaryUtils::OperandPredicate20(m_destinationB);

		// SourceC
		code |= BinaryUtils::OperandPredicate23(m_sourceC, m_flags & Flags::NOT_C); 

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
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Predicate *m_sourceC = nullptr;
	Predicate *m_sourceD = nullptr;

	ComparisonOperator m_comparisonOperator;
	BooleanOperator m_booleanOperator;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(ISETPInstruction)

}
}
