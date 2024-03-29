#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Operand.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Maxwell {

class I2IInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		NEG  = 0x0000200000000000,
		ABS  = 0x0002000000000000,
		SAT  = 0x0004000000000000
	};

	enum class SourceType : std::uint64_t {
		S8  = 0x0000000000002000,
		S16 = 0x0000000000002400,
		S32 = 0x0000000000002800,

		U8  = 0x0000000000000000,
		U16 = 0x0000000000000400,
		U32 = 0x0000000000000800
	};

	enum class DestinationType : std::uint64_t {
		S8  = 0x0000000000001000,
		S16 = 0x0000000000001100,
		S32 = 0x0000000000001200,

		U8  = 0x0000000000000000,
		U16 = 0x0000000000000100,
		U32 = 0x0000000000000200
	};

	SASS_FLAGS_FRIEND()

	I2IInstruction(Register *destination, Composite *source, DestinationType destinationType, SourceType sourceType, Flags flags = Flags::None)
		: m_destination(destination), m_source(source), m_destinationType(destinationType), m_sourceType(sourceType), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Composite *GetSource() const { return m_source; }
	Composite *GetSource() { return m_source; }
	void SetSource(Composite *source) { m_source = source; }

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
	
	std::string OpCode() const override { return "I2I"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_destinationType)
		{
			case DestinationType::S8: code += ".S8"; break;
			case DestinationType::S16: code += ".S16"; break;
			case DestinationType::S32: code += ".S32"; break;

			case DestinationType::U8: code += ".U8"; break;
			case DestinationType::U16: code += ".U16"; break;
			case DestinationType::U32: code += ".U32"; break;
		}
		switch (m_sourceType)
		{
			case SourceType::S8: code += ".S8"; break;
			case SourceType::S16: code += ".S16"; break;
			case SourceType::S32: code += ".S32"; break;

			case SourceType::U8: code += ".U8"; break;
			case SourceType::U16: code += ".U16"; break;
			case SourceType::U32: code += ".U32"; break;
		}
		if (m_flags & Flags::SAT)
		{
			code += ".SAT";
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
		return BinaryUtils::OpCodeComposite(0x5ce0000000000000, m_source);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		auto code = BinaryUtils::OpModifierFlags(m_sourceType) |
		            BinaryUtils::OpModifierFlags(m_destinationType) |
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

	InstructionClass GetInstructionClass() const override { return InstructionClass::SpecialFunction; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Composite *m_source = nullptr;

	DestinationType m_destinationType;
	SourceType m_sourceType;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(I2IInstruction)

}
}
