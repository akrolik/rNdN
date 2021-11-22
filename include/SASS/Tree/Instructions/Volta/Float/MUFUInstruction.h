#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class MUFUInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		NEG  = (1 << 0),
		ABS  = (1 << 1)
	};

	SASS_FLAGS_FRIEND()

	enum class Function : std::uint64_t {
		COS    = 0x0,
		SIN    = 0x1,
		EX2    = 0x2,
		LG2    = 0x3,
		RCP    = 0x4,
		RSQ    = 0x5,
		RCP64H = 0x6,
		RSQ64H = 0x7,
		SQRT   = 0x8
	};

	MUFUInstruction(Register *destination, Register *source, Function function, Flags flags = Flags::None)
		: m_destination(destination), m_source(source), m_function(function), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Register *GetSource() const { return m_source; }
	Register *GetSource() { return m_source; }
	void SetSource(Register *source) { m_source = source; }

	Function GetFunction() const { return m_function; }
	void SetFunction(Function function) { m_function = function; }

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
	
	std::string OpCode() const override { return "MUFU"; }

	std::string OpModifiers() const override
	{
		switch (m_function)
		{
			case Function::COS: return ".COS";
			case Function::SIN: return ".SIN";
			case Function::EX2: return ".EX2";
			case Function::LG2: return ".LG2";
			case Function::RCP: return ".RCP";
			case Function::RSQ: return ".RSQ";
			case Function::RCP64H: return ".RCP64H";
			case Function::RSQ64H: return ".RSQ64H";
			case Function::SQRT: return ".SQRT";
		}
		return "";
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
		if (m_flags & Flags::ABS)
		{
			code += "|";
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCode(m_source,
			0x308, // Register
			0x908, // Immediate
			0xb08  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// Source
		code |= BinaryUtils::OperandComposite(m_source);

		// Flags (Source)
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

		// Function
		code |= BinaryUtils::Format(m_function, 10, 0xf);

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

	Function m_function;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(MUFUInstruction)

}
}
