#pragma once

#include "PTX/Statements/InstructionStatement.h"

namespace PTX {

class ReturnInstruction : public InstructionStatement
{
public:
	ReturnInstruction(bool uniform = false) : m_uniform(uniform) {}

	std::string OpCode()
	{
		if (m_uniform)
		{
			return "ret.uni";
		}
		return "ret";
	}

	std::string Operands()
	{
		return "";
	}

private:
	bool m_uniform = false;
};

}
