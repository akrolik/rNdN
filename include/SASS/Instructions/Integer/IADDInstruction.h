#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class IADDInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		SAT   = 0x0004000000000000,
		X     = 0x0000080000000000,
		CC    = 0x0000800000000000,
		NEG_I = 0x0100000000000000,
		NEG_A = 0x0002000000000000,
		NEG_B = 0x0001000000000000
	};

	SASS_FLAGS_FRIEND()

	IADDInstruction(Register *destination, Register *sourceA, Composite *sourceB, Flags flags = Flags::None)
		: PredicatedInstruction({destination, sourceA, sourceB}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_flags(flags) {}

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

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "IADD"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::SAT)
		{
			code += ".SAT";
		}
		if (m_flags & Flags::X)
		{
			code += ".X";
		}
		return code;
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
		if (m_flags & Flags::NEG_A)
		{
			code += "-";
		}
		code += m_sourceA->ToString();
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_I || m_flags & Flags::NEG_B)
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::NEG_I && m_flags & Flags::NEG_B)
		{
			code += ".NEG";
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c10000000000000, m_sourceB);
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

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::x32; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(IADDInstruction)

}
