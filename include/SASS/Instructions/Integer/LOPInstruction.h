#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Register.h"

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

	//TODO: Enable Z constructor
	enum class PredicateZ : std::uint64_t {
		None = 0x0007000000000000,
		Z    = 0x0000200000000000,
		NZ   = 0x0000300000000000
	};

	SASS_FLAGS_FRIEND()

	LOPInstruction(Register *destination, Register *sourceA, Composite *sourceB, BooleanOperator booleanOperator, Flags flags = Flags::None)
		: PredicatedInstruction({destination, sourceA, sourceB}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_booleanOperator(booleanOperator), m_flags(flags) {}

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

	BooleanOperator GetBooleanOperator() const { return m_booleanOperator; }
	void SetBooleanOperator(BooleanOperator booleanOperator) { m_booleanOperator = booleanOperator; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "LOP"; }

	std::string OpModifiers() const override
	{
		std::string code;
		//TODO: Z operand
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

		//TODO: Z operand
		
		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
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

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c40000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		//TODO: Z operand
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB);
	}

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;

	BooleanOperator m_booleanOperator;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(LOPInstruction)

}
