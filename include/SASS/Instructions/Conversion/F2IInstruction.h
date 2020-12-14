#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Operand.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class F2IInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		//TODO: Do we need NEG_I?
		FTZ  = 0x0000100000000000,
		NEG  = 0x0000200000000000,
		ABS  = 0x0002000000000000,

		H0   = 0x0000000000000000,
		H1   = 0x0000040000000000,
		B0   = 0x0000000000000000,
		B1   = 0x0000020000000000,
		B2   = 0x0000040000000000,
		B3   = 0x0000060000000000
	};

	enum class DestinationType : std::uint64_t {
		F32 = 0x0000000000000200,
		F64 = 0x0000000000000300,
	};

	enum class SourceType : std::uint64_t {
		S8  = 0x0000000000002000,
		S16 = 0x0000000000002400,
		S32 = 0x0000000000002800,
		S64 = 0x0000000000002c00,

		U8  = 0x0000000000000000,
		U16 = 0x0000000000000400,
		U32 = 0x0000000000000800,
		U64 = 0x0000000000000c00
	};

	enum class Round : std::uint64_t {
		ROUND = 0x0000000000000000,
		FLOOR = 0x0000008000000000,
		CEIL  = 0x0000010000000000,
		TRUNC = 0x0000018000000000
	};

	SASS_FLAGS_FRIEND()

	F2IInstruction(Register *destination, Composite *source, DestinationType destinationType, SourceType sourceType, Round round = Round::ROUND, Flags flags = Flags::None)
		: PredicatedInstruction({destination, source}), m_destination(destination), m_source(source), m_destinationType(destinationType), m_sourceType(sourceType), m_round(round), m_flags(flags) {}

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
			case DestinationType::F32: code += ".F32"; break;
			case DestinationType::F64: code += ".F64"; break;
		}
		switch (m_sourceType)
		{
			case SourceType::S8: code += ".S8"; break;
			case SourceType::S16: code += ".S16"; break;
			case SourceType::S32: code += ".S32"; break;
			case SourceType::S64: code += ".S64"; break;

			case SourceType::U8: code += ".U8"; break;
			case SourceType::U16: code += ".U16"; break;
			case SourceType::U32: code += ".U32"; break;
			case SourceType::U64: code += ".U64"; break;
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
		if (m_flags & Flags::NEG)
		{
			code += "-";
		}
		if (m_flags & Flags::ABS)
		{
			code += "|";
		}
		code += m_source->ToString();
		if (m_flags & Flags::H0)
		{
			code += ".H0";
		}
		if (m_flags & Flags::H1)
		{
			code += ".H1";
		}
		if (m_flags & Flags::B0)
		{
			code += ".B0";
		}
		if (m_flags & Flags::B1)
		{
			code += ".B1";
		}
		if (m_flags & Flags::B2)
		{
			code += ".B2";
		}
		if (m_flags & Flags::B3)
		{
			code += ".B3";
		}
		if (m_flags & Flags::ABS)
		{
			code += "|";
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
		return BinaryUtils::OpModifierFlags(m_sourceType) |
		       BinaryUtils::OpModifierFlags(m_destinationType) |
		       BinaryUtils::OpModifierFlags(m_round) |
		       BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandComposite(m_source);
	}

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
