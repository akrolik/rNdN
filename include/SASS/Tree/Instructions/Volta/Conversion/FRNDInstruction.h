#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class FRNDInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		FTZ  = (1 << 0),
		NEG  = (1 << 2),
		ABS  = (1 << 3)
	};

	enum class Round : std::uint64_t {
		ROUND = 0x0,
		FLOOR = 0x1,
		CEIL  = 0x2,
		TRUNC = 0x3
	};

	enum class Type : std::uint64_t {
		F32 = 0x2,
		F64 = 0x3
	};

	SASS_FLAGS_FRIEND()

	FRNDInstruction(Register *destination, Composite *source, Type type, Round round = Round::ROUND, Flags flags = Flags::None)
		: m_destination(destination), m_source(source), m_type(type), m_round(round), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Composite *GetSource() const { return m_source; }
	Composite *GetSource() { return m_source; }
	void SetSource(Composite *source) { m_source = source; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }
 
	Round GetRound() const { return m_round; }
	void SetRound(Round round) { m_round = round; }

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
	
	std::string OpCode() const override { return "FRND"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::FTZ)
		{
			code += ".FTZ";
		}
		switch (m_type)
		{
			// case Type::F32: code += ".F32"; break;
			case Type::F64: code += ".F64"; break;
		}
		switch (m_round)
		{
			case Round::ROUND: code += ""; break;
			case Round::FLOOR: code += ".FLOOR"; break;
			case Round::CEIL: code += ".CEIL"; break;
			case Round::TRUNC: code += ".TRUNC"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;
		
		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		if (m_flags & Flags::NEG)
		{
			code += "-";
		}
		if (m_flags & Flags::ABS)
		{
			code += "|";
		}
		code += m_source->ToString();
		if (m_flags & Flags::ABS)
		{
			code += "|";
		}
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		if (m_type == Type::F64)
		{
			return BinaryUtils::OpCode(m_source,
				0x313, // Register
				0x913, // Immediate
				0xb13  // Constant
			);
		}
		return BinaryUtils::OpCode(m_source,
			0x307, // Register
			0x907, // Immediate
			0xb07  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// Source
		code |= BinaryUtils::OperandComposite(m_source, m_flags & Flags::NEG, m_flags & Flags::ABS);

		// Flags (constant/register Source)
		if (m_source->GetKind() != Operand::Kind::Immediate)
		{
			code |= BinaryUtils::FlagBit(m_flags & Flags::ABS, 62);
			code |= BinaryUtils::FlagBit(m_flags & Flags::NEG, 63);
		}

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Source Type
		code |= BinaryUtils::Format(m_type, 20, 0x3);

		// Destination Type
		code |= BinaryUtils::Format(m_type, 11, 0x3);

		// Flags
		code |= BinaryUtils::FlagBit(m_flags & Flags::FTZ, 16);

		// Rounding
		code |= BinaryUtils::Format(m_round, 14, 0x3);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override
	{
		if (m_type == Type::F64)
		{
			return InstructionClass::DoublePrecision;
		}
		return InstructionClass::SpecialFunction;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Composite *m_source = nullptr;

	Type m_type;
	Round m_round = Round::ROUND;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(FRNDInstruction)

}
}
