#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Predicate.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class DMNMXInstruction : public Instruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		NEG_I = 0x0100000000000000,
		NEG_A = 0x0001000000000000,
		NEG_B = 0x0000200000000000,
		ABS_A = 0x0000400000000000,
		ABS_B = 0x0002000000000000,
		NOT_C = 0x0000040000000000
	};

	SASS_FLAGS_FRIEND()

	DMNMXInstruction(const Register *destination, const Register *sourceA, const Composite *sourceB, const Predicate *sourceC, Flags flags = Flags::None)
		: Instruction({destination, sourceA, sourceB, sourceC}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	void SetDestination(const Register *destination) { m_destination = destination; }

	const Register *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const Composite *sourceB) { m_sourceB = sourceB; }

	const Predicate *GetSourceC() const { return m_sourceC; }
	void SetSourceC(const Predicate *sourceC) { m_sourceC = sourceC; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting
	
	std::string OpCode() const override { return "DMNMX"; }

	std::string Operands() const override
	{
		std::string code;
		
		// Destination
		code += m_destination->ToString();
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
		return BinaryUtils::OpCodeComposite(0x5c50000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB) |
		       BinaryUtils::OperandPredicate39(m_sourceC);
	}

private:
	const Register *m_destination = nullptr;
	const Register *m_sourceA = nullptr;
	const Composite *m_sourceB = nullptr;
	const Predicate *m_sourceC = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(DMNMXInstruction)

}
