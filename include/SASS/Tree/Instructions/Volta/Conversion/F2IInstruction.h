#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class F2IInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		FTZ  = (1 << 0),
		NTZ  = (1 << 1),
		NEG  = (1 << 2),
		ABS  = (1 << 3)
	};

	enum class Round : std::uint64_t {
		ROUND = 0x0,
		FLOOR = 0x1,
		CEIL  = 0x2,
		TRUNC = 0x3
	};

	enum class DestinationType : std::uint64_t {
		U8  = 0x0,
		U16 = 0x1,
		U32 = 0x2,
		U64 = 0x3,

		S8  = 0x4,
		S16 = 0x5,
		S32 = 0x6,
		S64 = 0x7
	};

	enum class SourceType : std::uint64_t {
		F16 = 0x1,
		F32 = 0x2,
		F64 = 0x3
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

	DestinationType GetDestinationType() const { return m_destinationType; }
	void SetDestinationType(DestinationType destinationType) { m_destinationType = destinationType; }
 
	SourceType GetSourceType() const { return m_sourceType; }
	void SetSourceType(SourceType sourceType) { m_sourceType = sourceType; }

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
			case DestinationType::U8: code += ".U8"; break;
			case DestinationType::U16: code += ".U16"; break;
			case DestinationType::U32: code += ".U32"; break;
			case DestinationType::U64: code += ".U64"; break;

			case DestinationType::S8: code += ".S8"; break;
			case DestinationType::S16: code += ".S16"; break;
			case DestinationType::S32: code += ".S32"; break;
			case DestinationType::S64: code += ".S64"; break;
		}
		switch (m_sourceType)
		{
			case SourceType::F16: code += ".F16"; break;
			case SourceType::F32: code += ".F32"; break;
			case SourceType::F64: code += ".F64"; break;
		}
		switch (m_round)
		{
			case Round::ROUND: code += ""; break;
			case Round::FLOOR: code += ".FLOOR"; break;
			case Round::CEIL: code += ".CEIL"; break;
			case Round::TRUNC: code += ".TRUNC"; break;
		}
		if (m_flags & Flags::NTZ)
		{
			code += ".NTZ";
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
		if (m_sourceType == SourceType::F64 ||
			m_destinationType == DestinationType::S64 ||
			m_destinationType == DestinationType::U64)
		{
			return BinaryUtils::OpCode(m_source,
				0x311, // Register
				0x911, // Immediate
				0xb11  // Constant
			);
		}
		return BinaryUtils::OpCode(m_source,
			0x305, // Register
			0x905, // Immediate
			0xb05  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// Source
		code |= BinaryUtils::OperandComposite(m_source);

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
		code |= BinaryUtils::Format(m_sourceType, 20, 0x3);

		// Destination Type
		code |= BinaryUtils::Format(m_destinationType, 11, 0x3);
		code |= BinaryUtils::FlagBit(m_destinationType >= DestinationType::S8, 8);

		// Flags (FTZ/NTZ)
		code |= BinaryUtils::FlagBit(m_flags & Flags::NTZ, 13);
		code |= BinaryUtils::FlagBit(m_flags & Flags::FTZ, 16);

		// Rounding
		code |= BinaryUtils::Format(m_round, 14, 0x3);

		return code;
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
