#pragma once

#include "SASS/Instructions/Instruction.h"
#include "SASS/Operands/Address.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class LDGInstruction : public Instruction
{
public:
	enum Flags {
		None  = 0,
		E     = (1 << 0)
	};

	friend Flags operator|(Flags a, Flags b);
	friend Flags operator&(Flags a, Flags b);

	enum class Type {
		U8,
		S8,
		U16,
		S16,
		I32,
		I64,
		I128
	};

	static std::string TypeString(Type type)
	{
		switch (type)
		{
			case Type::U8:
				return ".U8";
			case Type::S8:
				return ".S8";
			case Type::U16:
				return ".U16";
			case Type::S16:
				return ".S16";
			case Type::I32:
				return ".32";
			case Type::I64:
				return ".64";
			case Type::I128:
				return ".128";
		}
		return ".<unknown>";
	}

	LDGInstruction(const Register *destination, const Address *source, Type type, Flags flags = Flags::None) : Instruction({destination, source}), m_destination(destination), m_source(source), m_type(type), m_flags(flags) {}

	std::string OpCode() const override { return "LDG"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::E)
		{
			code += ".E";
		}
		code += TypeString(m_type);
		return code;
	}

private:
	const Register *m_destination = nullptr;
	const Address *m_source = nullptr;

	Type m_type;
	Flags m_flags = Flags::None;
};

inline LDGInstruction::Flags operator&(LDGInstruction::Flags a, LDGInstruction::Flags b)
{
	return static_cast<LDGInstruction::Flags>(static_cast<int>(a) & static_cast<int>(b));
}

inline LDGInstruction::Flags operator|(LDGInstruction::Flags a, LDGInstruction::Flags b)
{
	return static_cast<LDGInstruction::Flags>(static_cast<int>(a) | static_cast<int>(b));
}

}
