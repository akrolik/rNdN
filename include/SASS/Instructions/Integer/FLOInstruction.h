#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class FLOInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		NEG_I = 0x0100000000000000
	};

	SASS_FLAGS_FRIEND()

	FLOInstruction(Register *destination, Composite *source, Flags flags = Flags::None)
		: PredicatedInstruction({destination, source}), m_destination(destination), m_source(source), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Composite *GetSource() const { return m_source; }
	Composite *GetSource() { return m_source; }
	void SetSource(Composite *source) { m_source = source; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "FLO"; }

	std::string OpModifiers() const override
	{
		return ".U32";
	}

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
		return BinaryUtils::OpCodeComposite(0x5c30000000000000, m_source);
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

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Composite *m_source = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(FLOInstruction)

}
