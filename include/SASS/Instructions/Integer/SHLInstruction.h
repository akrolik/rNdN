#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class SHLInstruction : public Instruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		W     = 0x0000008000000000, 
		NEG_I = 0x0100000000000000
	};

	SASS_FLAGS_FRIEND()

	SHLInstruction(const Register *destination, const Register *sourceA, const Composite *sourceB, Flags flags = Flags::None)
		: Instruction({destination, sourceA, sourceB}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	void SetDestination(const Register *destination) { m_destination = destination; }

	const Register *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const Composite *sourceB) { m_sourceB = sourceB; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "SHL"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::W)
		{
			return ".W";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
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

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c48000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB);
	}

private:
	const Register *m_destination = nullptr;
	const Register *m_sourceA = nullptr;
	const Composite *m_sourceB = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(SHLInstruction)

}

