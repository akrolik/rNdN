#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/I32Immediate.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class IADD32IInstruction : public Instruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		X     = 0x0020000000000000,
		CC    = 0x0010000000000000,
		NEG_A = 0x0100000000000000
	};

	SASS_FLAGS_FRIEND()

	IADD32IInstruction(const Register *destination, const Register *sourceA, const I32Immediate *sourceB, Flags flags = Flags::None)
		: Instruction({destination, sourceA, sourceB}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	void SetDestination(const Register *destination) { m_destination = destination; }

	const Register *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const Register *sourceA) { m_sourceA = sourceA; }

	const I32Immediate *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const I32Immediate *sourceB) { m_sourceB = sourceB; }
	
	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "IADD32I"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::X)
		{
			return ".X";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		if (m_flags & Flags::CC)
		{
			code += ".CC";
		}
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		if (m_flags & Flags::NEG_A)
		{
			code += "-";
		}
		code += ", ";

		// SourceB
		code += m_sourceB->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x1c00000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandLiteral20W32(m_sourceB);
	}

private:
	const Register *m_destination = nullptr;
	const Register *m_sourceA = nullptr;
	const I32Immediate *m_sourceB = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(IADD32IInstruction)

}