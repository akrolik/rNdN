#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Address.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class REDInstruction : public Instruction
{
public:
	enum Flags : std::uint64_t {
		None = 0x0,
		E    = 0x0001000000000000
	};

	SASS_FLAGS_FRIEND()

	enum class Type : std::uint64_t {
	     None = 0x0,
	     S32  = 0x0000000000100000,
	     U64  = 0x0000000000200000,
	     F32  = 0x0000000000300000,
	     F16  = 0x0000000000400000,
	     S64  = 0x0000000000500000
	};

	enum class Mode : std::uint64_t {
	     ADD = 0x0000000000000000,
	     MIN = 0x0000000000800000,
	     MAX = 0x0000000001000000,
	     INC = 0x0000000001800000,
	     DEC = 0x0000000002000000,
	     AND = 0x0000000002800000,
	     OR  = 0x0000000003000000,
	     XOR = 0x0000000003800000
	};

	REDInstruction(const Address *destination, const Register *source, Type type, Mode mode, Flags flags = Flags::None) : Instruction({destination, source}), m_destination(destination), m_source(source), m_type(type), m_mode(mode), m_flags(flags) {}

	// Properties

	const Address *GetDestination() const { return m_destination; }
	void SetDestination(const Address *destination) { m_destination = destination; }

	const Register *GetSource() const { return m_source; }
	void SetSource(const Register *source) { m_source = source; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "RED"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::E)
		{
			code += ".E";
		}
		switch (m_mode)
		{
			case Mode::ADD: code += ".ADD"; break;
			case Mode::MIN: code += ".MIN"; break;
			case Mode::MAX: code += ".MAX"; break;
			case Mode::INC: code += ".INC"; break;
			case Mode::DEC: code += ".DEC"; break;
			case Mode::AND: code += ".AND"; break;
			case Mode::OR: code += ".OR"; break;
			case Mode::XOR: code += ".XOR"; break;
		}
		switch (m_type)
		{
			case Type::S32: code += ".S32"; break;
			case Type::U64: code += ".U64"; break;
			case Type::F32: code += ".F32.FTZ.RN"; break;
			case Type::F16: code += ".F16x2.FTZ.RN"; break;
			case Type::S64: code += ".S64"; break;
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
		return 0xebf8000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags) |
		       BinaryUtils::OpModifierFlags(m_type) |
		       BinaryUtils::OpModifierFlags(m_mode);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_source) |
		       BinaryUtils::OperandRegister8(m_destination->GetBase()) |
		       BinaryUtils::OperandAddress28W20(m_destination->GetOffset());
	}

private:
	const Address *m_destination = nullptr;
	const Register *m_source = nullptr;

	Type m_type;
	Mode m_mode;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(REDInstruction)

}
