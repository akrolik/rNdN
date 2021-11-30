#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class F2FInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		FTZ  = (1 << 0),
		NEG  = (1 << 1),
		ABS  = (1 << 2)
	};

	enum class Round : std::uint64_t {
		RN = 0x0,
		RM = 0x1,
		RP = 0x2,
		RZ = 0x3
	};

	enum class DestinationType : std::uint64_t {
		F16 = 0x1,
		F32 = 0x2,
		F64 = 0x3
	};

	enum class SourceType : std::uint64_t {
		F16 = 0x1,
		F32 = 0x2,
		F64 = 0x3
	};

	SASS_FLAGS_FRIEND()

	F2FInstruction(Register *destination, Composite *source, DestinationType destinationType, SourceType sourceType, Round round = Round::RN, Flags flags = Flags::None)
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
	
	std::string OpCode() const override { return "F2F"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::FTZ)
		{
			code += ".FTZ";
		}
		switch (m_destinationType)
		{
			case DestinationType::F16: code += ".F16"; break;
			case DestinationType::F32: code += ".F32"; break;
			case DestinationType::F64: code += ".F64"; break;
		}
		switch (m_sourceType)
		{
			case SourceType::F16: code += ".F16"; break;
			case SourceType::F32: code += ".F32"; break;
			case SourceType::F64: code += ".F64"; break;
		}
		switch (m_round)
		{
			case Round::RN: code += ""; break;
			case Round::RM: code += ".RM"; break;
			case Round::RP: code += ".RP"; break;
			case Round::RZ: code += ".RZ"; break;
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
		if (m_sourceType == SourceType::F64 || m_destinationType == DestinationType::F64)
		{
			return BinaryUtils::OpCode(m_source,
				0x310, // Register
				0x910, // Immediate
				0xb10  // Constant
			);
		}
		return BinaryUtils::OpCode(m_source,
			0x304, // Register
			0x904, // Immediate
			0xb04  // Constant
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
		code |= BinaryUtils::Format(m_sourceType, 20, 0x3);

		// Destination Type
		code |= BinaryUtils::Format(m_destinationType, 11, 0x3);

		// Flags (FTZ/NTZ)
		code |= BinaryUtils::FlagBit(m_flags & Flags::FTZ, 16);

		// Rounding
		code |= BinaryUtils::Format(m_round, 14, 0x3);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override
	{
		if (m_sourceType == SourceType::F64 || m_destinationType == DestinationType::F64)
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
	Round m_round = Round::RN;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(F2FInstruction)

}
}
