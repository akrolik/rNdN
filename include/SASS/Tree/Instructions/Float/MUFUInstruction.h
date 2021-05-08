#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class MUFUInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		NEG  = 0x0001000000000000
	};

	SASS_FLAGS_FRIEND()

	enum class Function : std::uint64_t {
		COS    = 0x0000000000000000,
		SIN    = 0x0000000000100000,
		EX2    = 0x0000000000200000,
		LG2    = 0x0000000000300000,
		RCP    = 0x0000000000400000,
		RSQ    = 0x0000000000500000,
		RCP64H = 0x0000000000600000,
		RSQ64H = 0x0000000000700000
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
		code += m_source->ToString();
		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x5080000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_function) | BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_source);
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::qtr; }

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
