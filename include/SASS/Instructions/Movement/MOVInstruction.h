#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Operand.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class MOVInstruction : public Instruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		NEG_I = 0x0100000000000000
	};

	SASS_FLAGS_FRIEND()

	MOVInstruction(const Register *destination, const Composite *source, Flags flags = Flags::None)
		: Instruction({destination, source}), m_destination(destination), m_source(source), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	void SetDestination(const Register *destination) { m_destination = destination; }

	const Composite *GetSource() const { return m_source; }
	void SetSource(const Composite *source) { m_source = source; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting
	
	std::string OpCode() const override { return "MOV"; }

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// Source
		if (m_flags & Flags::NEG_I)
		{
			code += "-";
		}
		code += m_source->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c98078000000000, m_source);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandComposite(m_source);
	}

private:
	const Register *m_destination = nullptr;
	const Composite *m_source = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(MOVInstruction)

}
