#pragma once

#include "SASS/Instructions/XMADInstruction.h"

namespace SASS {

class XMADInstruction : public Instruction
{
public:
	enum Flags {
		None  = 0,
		MRG   = (1 << 0),
		PSL   = (1 << 1),
		CBCC  = (1 << 2),
		H1A   = (1 << 3),
		H1B   = (1 << 4),
		H1C   = (1 << 5)
	};

	friend Flags operator|(Flags a, Flags b);
	friend Flags operator&(Flags a, Flags b);

	XMADInstruction(const Register *destination, const Operand *sourceA, const Operand *sourceB, const Operand *sourceC, Flags flags = Flags::None) : Instruction({destination, sourceA, sourceB, sourceC}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_flags(flags) {}

	std::string OpCode() const override { return "XMAD"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_flags & Flags::MRG)
		{
			code += ".MRG";
		}
		if (m_flags & Flags::PSL)
		{
			code += ".PSL";
		}
		if (m_flags & Flags::CBCC)
		{
			code += ".CBCC";
		}
		return code;
	}

	std::string OperandModifiers(unsigned int index) const override
	{
		std::string code;
		if (index == 1)
		{
			if (m_flags & Flags::H1A)
			{
				code += ".H1";
			}
		}
		else if (index == 2)
		{
			if (m_flags & Flags::H1B)
			{
				code += ".H1";
			}
		}
		else if (index == 3)
		{
			if (m_flags & Flags::H1C)
			{
				code += ".H1";
			}
		}
		return code;
	}

private:
	const Register *m_destination = nullptr;
	const Operand *m_sourceA = nullptr;
	const Operand *m_sourceB = nullptr;
	const Operand *m_sourceC = nullptr;

	Flags m_flags = Flags::None;
};

inline XMADInstruction::Flags operator&(XMADInstruction::Flags a, XMADInstruction::Flags b)
{
	return static_cast<XMADInstruction::Flags>(static_cast<int>(a) & static_cast<int>(b));
}

inline XMADInstruction::Flags operator|(XMADInstruction::Flags a, XMADInstruction::Flags b)
{
	return static_cast<XMADInstruction::Flags>(static_cast<int>(a) | static_cast<int>(b));
}

}
