#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Operand.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Maxwell {

class F2IInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		FTZ  = 0x0000100000000000,
		NEG  = 0x0000200000000000,
		ABS  = 0x0002000000000000
	};

	enum class DestinationType : std::uint64_t {
		S8  = 0x0000000000001000,
		S16 = 0x0000000000001100,
		S32 = 0x0000000000001200,
		S64 = 0x0000000000001300,

		U8  = 0x0000000000000000,
		U16 = 0x0000000000000100,
		U32 = 0x0000000000000200,
		U64 = 0x0000000000000300
	};

	enum class SourceType : std::uint64_t {
		F32 = 0x0000000000000800,
		F64 = 0x0000000000000c00
	};

	enum class Round : std::uint64_t {
		ROUND = 0x0000000000000000,
		FLOOR = 0x0000008000000000,
		CEIL  = 0x0000010000000000,
		TRUNC = 0x0000018000000000
	};

	SASS_FLAGS_FRIEND()

	F2IInstruction(Register *destination, Composite *source, DestinationType destinationType, SourceType sourceType, Round round = Round::ROUND, Flags flags = Flags::None)
		: m_destination(destination), m_source(source), m_destinationType(destinationType), m_sourceType(sourceType), m_round(round), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Composite *GetSource() const { return m_source; }
	Composite *GetSource() { return m_source; }
	void SetSource(Composite *source) { m_source = source; }

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
	
	std::string OpCode() const override { return "F2I"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::FTZ)
		{
			code += ".FTZ";
		}
		switch (m_destinationType)
		{
			case DestinationType::S8: code += ".S8"; break;
			case DestinationType::S16: code += ".S16"; break;
			case DestinationType::S32: code += ".S32"; break;
			case DestinationType::S64: code += ".S64"; break;

			case DestinationType::U8: code += ".U8"; break;
			case DestinationType::U16: code += ".U16"; break;
			case DestinationType::U32: code += ".U32"; break;
			case DestinationType::U64: code += ".U64"; break;
		}
		switch (m_sourceType)
		{
			case SourceType::F32: code += ".F32"; break;
			case SourceType::F64: code += ".F64"; break;
		}
		switch (m_round)
		{
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

		// Source
		if ((m_flags & Flags::NEG) && !m_source->GetOpModifierNegate())
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
		if ((m_flags & Flags::NEG) && m_source->GetOpModifierNegate())
		{
			code += ".NEG";
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5cb0000000000000, m_source);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		auto code = BinaryUtils::OpModifierFlags(m_sourceType) |
		            BinaryUtils::OpModifierFlags(m_destinationType) |
		            BinaryUtils::OpModifierFlags(m_round) |
		            BinaryUtils::OpModifierFlags(m_flags);

		if (m_source->GetOpModifierNegate())
		{
			code |= 0x0100000000000000;
		}
		return code;
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandComposite(m_source);
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override
	{
		if (m_sourceType == SourceType::F64)
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

	DestinationType m_destinationType;
	SourceType m_sourceType;

	Round m_round = Round::ROUND;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(F2IInstruction)

}
}
