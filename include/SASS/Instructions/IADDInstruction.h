#pragma once

#include "SASS/Instructions/Instruction.h"
#include "SASS/Operands/Operand.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class IADDInstruction : public Instruction
{
public:
	enum Flags {
		None  = 0,
		CC    = (1 << 0),
		X     = (1 << 1),
		SAT   = (1 << 2),
		NEG0  = (1 << 3),
		NEG1  = (1 << 4),
		NEG2  = (1 << 5)
	};

	friend Flags operator|(Flags a, Flags b);
	friend Flags operator&(Flags a, Flags b);

	IADDInstruction(const Register *destination, const Operand *sourceA, const Operand *sourceB, Flags flags = Flags::None) : Instruction({destination, sourceA, sourceB}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_flags(flags) {}

	std::string OpCode() const override { return "IADD"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::X)
		{
			code += ".X";
		}
		return code;
	}

	std::string OperandModifiers(unsigned int index) const override
	{
		std::string code;
		if (index == 0)
		{
			if (m_flags & Flags::CC)
			{
				code += ".CC";
			}
		}
		return code;
	}

private:
	const Register *m_destination = nullptr;
	const Operand *m_sourceA = nullptr;
	const Operand *m_sourceB = nullptr;

	Flags m_flags = Flags::None;
};

inline IADDInstruction::Flags operator&(IADDInstruction::Flags a, IADDInstruction::Flags b)
{
	return static_cast<IADDInstruction::Flags>(static_cast<int>(a) & static_cast<int>(b));
}

inline IADDInstruction::Flags operator|(IADDInstruction::Flags a, IADDInstruction::Flags b)
{
	return static_cast<IADDInstruction::Flags>(static_cast<int>(a) | static_cast<int>(b));
}

}
