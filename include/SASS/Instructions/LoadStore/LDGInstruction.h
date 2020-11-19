#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Address.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class LDGInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0x0,
		E    = 0x0000200000000000
	};

	SASS_FLAGS_FRIEND()

	enum class Cache : std::uint64_t {
		None = 0x0,
		CG   = 0x0000400000000000,
		CS   = 0x0000800000000000,
		CV   = 0x0000c00000000000
	};

	enum class Type : std::uint64_t {
		U8   = 0x0000000000000000,
		S8   = 0x0001000000000000,
		U16  = 0x0002000000000000,
		S16  = 0x0003000000000000,
		I32  = 0x0004000000000000,
		I64  = 0x0005000000000000,
		I128 = 0x0006000000000000
	};

	LDGInstruction(const Register *destination, const Address *source, Type type, Cache cache = Cache::None, Flags flags = Flags::None)
		: PredicatedInstruction({destination, source}), m_destination(destination), m_source(source), m_type(type), m_cache(cache), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	void SetDestination(const Register *destination) { m_destination = destination; }

	const Address *GetSource() const { return m_source; }
	void SetSource(const Address *source) { m_source = source; }

	Type GetType() const { return m_type; }
	void SetType(Type type) { m_type = type; }

	Cache GetCache() const { return m_cache; }
	void SetCache(Cache cache) { m_cache = cache; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "LDG"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_cache)
		{
			case Cache::CG: code += ".CG"; break;
			case Cache::CS: code += ".CS"; break;
			case Cache::CV: code += ".CV"; break;
		}
		if (m_flags & Flags::E)
		{
			code += ".E";
		}
		switch (m_type)
		{
			case Type::U8: code += ".U8"; break;
			case Type::S8: code += ".S8"; break;
			case Type::U16: code += ".U16"; break;
			case Type::S16: code += ".S16"; break;
			case Type::I32: code += ".I32"; break;
			case Type::I64: code += ".I64"; break;
			case Type::I128: code += ".I128"; break;
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
		return 0xeed0000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags) |
		       BinaryUtils::OpModifierFlags(m_type) |
		       BinaryUtils::OpModifierFlags(m_cache);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_source->GetBase()) |
		       BinaryUtils::OperandAddress20W24(m_source->GetOffset());
	}

private:
	const Register *m_destination = nullptr;
	const Address *m_source = nullptr;

	Cache m_cache = Cache::None;
	Type m_type;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(LDGInstruction)

}
