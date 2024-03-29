#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Address.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Maxwell {

class LDSInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0x0,
		U    = 0x0000100000000000,
		E    = 0x0000200000000000
	};

	SASS_FLAGS_FRIEND()

	enum class Type : std::uint64_t {
		U8   = 0x0000000000000000,
		S8   = 0x0001000000000000,
		U16  = 0x0002000000000000,
		S16  = 0x0003000000000000,
		X32  = 0x0004000000000000,
		X64  = 0x0005000000000000,
		X128 = 0x0006000000000000
	};

	LDSInstruction(Register *destination, Address *source, Type type, Flags flags = Flags::None)
		: m_destination(destination), m_source(source), m_type(type), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Address *GetSource() const { return m_source; }
	Address *GetSource() { return m_source; }
	void SetSource(Address *source) { m_source = source; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

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

	std::string OpCode() const override { return "LDS"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::E)
		{
			code += ".E";
		}
		if (m_flags & Flags::U)
		{
			code += ".U";
		}
		switch (m_type)
		{
			case Type::U8: code += ".U8"; break;
			case Type::S8: code += ".S8"; break;
			case Type::U16: code += ".U16"; break;
			case Type::S16: code += ".S16"; break;
			// case Type::X32: code += ".32"; break;
			case Type::X64: code += ".64"; break;
			case Type::X128: code += ".128"; break;
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
		code += m_source->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xef48000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags) |
		       BinaryUtils::OpModifierFlags(m_type);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_source->GetBase()) |
		       BinaryUtils::OperandAddress20W24(m_source->GetOffset());
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::SharedMemoryLoad; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Address *m_source = nullptr;

	Type m_type;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(LDSInstruction)

}
}
