#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

class FLOInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		INV   = (1 << 0),
		U32   = (1 << 1),
		SH    = (1 << 2)
	};

	SASS_FLAGS_FRIEND()

	FLOInstruction(Register *destination, Register *source, Flags flags = Flags::None)
		: m_destination(destination), m_source(source), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Register *GetSource() const { return m_source; }
	Register *GetSource() { return m_source; }
	void SetSource(Register *source) { m_source = source; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_source };
	}

	// Formatting

	std::string OpCode() const override { return "FLO"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::U32)
		{
			code += ".U32";
		}
		if (m_flags & Flags::SH)
		{
			code += ".SH";
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// Source
		if (m_flags & Flags::INV)
		{
			code += "~";
		}
		code += m_source->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x300;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// Source
		code |= BinaryUtils::OperandRegister32(m_source);

		// Flags
		code |= BinaryUtils::FlagBit(m_flags & Flags::INV, 63);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Flags
		code |= BinaryUtils::FlagBit(!(m_flags & Flags::U32), 9);
		code |= BinaryUtils::FlagBit(m_flags & Flags::SH, 10);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::SpecialFunction; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_source = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(FLOInstruction)

}
}
