#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/UniformModifier.h"

namespace PTX {

class ReturnInstruction : public InstructionStatement, public UniformModifier
{
public:
	ReturnInstruction(bool uniform = false) : UniformModifier(uniform) {}

	std::string OpCode() const
	{
		if (m_uniform)
		{
			return "ret.uni";
		}
		return "ret";
	}

	std::string Operands() const
	{
		return "";
	}
};

}
