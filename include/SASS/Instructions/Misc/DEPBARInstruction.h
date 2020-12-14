#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/I8Immediate.h"

namespace SASS {

class DEPBARInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		LE   = 0x0000000020000000
	};

	enum Barrier : std::uint64_t {
		SB0 = 0x0000000000000000,
		SB1 = 0x0000000004000000,
		SB2 = 0x0000000008000000,
		SB3 = 0x000000000c000000,
		SB4 = 0x0000000010000000,
		SB5 = 0x0000000014000000
	};

	SASS_FLAGS_FRIEND()

	DEPBARInstruction(Barrier barrier, I8Immediate *value, Flags flags = Flags::None) : m_barrier(barrier), m_value(value), m_flags(flags) {}

	// Properties

	Barrier GetBarrier() const { return m_barrier; }
	void SetBarrier(Barrier barrier) { m_barrier = barrier; }

	const I8Immediate *GetValue() const { return m_value; }
	I8Immediate *GetValue() { return m_value; }
	void SetValue(I8Immediate *value) { m_value = value; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "DEPBAR"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::LE)
		{
			return ".LE";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;

		// Barrier
		switch (m_barrier)
		{
			case Barrier::SB0: code += ".SB0"; break;
			case Barrier::SB1: code += ".SB1"; break;
			case Barrier::SB2: code += ".SB2"; break;
			case Barrier::SB3: code += ".SB3"; break;
			case Barrier::SB4: code += ".SB4"; break;
			case Barrier::SB5: code += ".SB5"; break;
		}
		code += ", ";

		// Literal
		code += m_value->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xf0f0000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OpModifierFlags(m_barrier) |
		       BinaryUtils::OperandLiteral20W6(m_value);
	}

private:
	Barrier m_barrier;
	I8Immediate *m_value = nullptr;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(DEPBARInstruction)

}
