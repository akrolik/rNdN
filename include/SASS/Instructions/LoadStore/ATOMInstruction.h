#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Address.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class ATOMInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0x0,
		E    = 0x0001000000000000
	};

	SASS_FLAGS_FRIEND()

	enum class Type : std::uint64_t {
		U32 = 0x0000000000000000,
		// S32 = 0x0002000000000000,
		S32 = 0x0002100000000000, //TODO: Temporarily disabled
		U64 = 0x0004000000000000,
		F32 = 0x0006000000000000,
		F16 = 0x0008000000000000,
		S64 = 0x000a000000000000,
		X64 = 0x0002000000000000
	};

	enum class Mode : std::uint64_t {
		ADD = 0x0000000000000000,
		MIN = 0x0010000000000000,
		MAX = 0x0020000000000000,
		INC = 0x0030000000000000,
		DEC = 0x0040000000000000,
		AND = 0x0050000000000000,
		OR  = 0x0060000000000000,
		XOR = 0x0070000000000000,
		EXCH = 0x0080000000000000,
		// CAS = 0x03f0000000000000
		CAS = 0x02f0000000000000 //TODO: Work out this
	};

	ATOMInstruction(Register *destination, Address *address, Register *source, Register *sourceC, Type type, Mode mode, Flags flags = Flags::None)
		: PredicatedInstruction({destination, address, source}), m_destination(destination), m_address(address), m_source(source), m_sourceC(sourceC), m_type(type), m_mode(mode), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Address *GetAddress() const { return m_address; }
	Address *GetAddress() { return m_address; }
	void SetAddress(Address *address) { m_address = address; }

	const Register *GetSource() const { return m_source; }
	Register *GetSource() { return m_source; }
	void SetSource(Register *source) { m_source = source; }

	const Register *GetSourceC() const { return m_sourceC; }
	Register *GetSourceC() { return m_sourceC; }
	void SetSourceC(Register *sourceC) { m_sourceC = sourceC; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "ATOM"; }

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
			case Mode::EXCH: code += ".EXCH"; break;
			case Mode::CAS: code += ".CAS"; break;
		}
		switch (m_type)
		{
			// case Type::U32: code += ".U32"; break; // Default
			case Type::S32: code += ".S32"; break;
			case Type::U64: code += ".U64"; break;
			case Type::F32: code += ".F32.FTZ.RN"; break;
			case Type::F16: code += ".F16x2.FTZ.RN"; break;
			case Type::S64: code += ".S64"; break;
			case Type::X64: code += ".64"; break;
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// Address
		code += m_address->ToString();
		code += ", ";

		// Source
		code += m_source->ToString();

		// Source C
		if (m_sourceC != nullptr)
		{
			code += ", ";
			code += m_sourceC->ToString();
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		//TODO: Figure this out
		if (m_mode == Mode::CAS)
		{
			return 0xec00000000000000;
		}
		return 0xed00000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags) |
		       BinaryUtils::OpModifierFlags(m_type) |
		       BinaryUtils::OpModifierFlags(m_mode);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_address->GetBase()) |
		       BinaryUtils::OperandAddress28W20(m_address->GetOffset()) |
		       BinaryUtils::OperandRegister20(m_source);

		// SourceC ignored for bit pattern, must be sequential
	}

private:
	Register *m_destination = nullptr;
	Address *m_address = nullptr;
	Register *m_source = nullptr;
	Register *m_sourceC = nullptr;

	Type m_type;
	Mode m_mode;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(ATOMInstruction)

}
