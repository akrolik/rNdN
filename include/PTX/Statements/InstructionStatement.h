#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Operands/Operand.h"

namespace PTX {

class InstructionStatement : public Statement
{
public:
	std::string ToString() const override
	{
		std::ostringstream code;
		code << OpCode();
		bool first = true;
		for (const auto& operand : Operands())
		{
			if (first)
			{
				code << " ";
				first = false;
			}
			else
			{
				code << ", ";
			}
			code << operand->ToString();
		}
		return code.str();
	}

	std::string Terminator() const override{ return ";"; }

	virtual std::string OpCode() const = 0;
	virtual std::vector<const Operand *> Operands() const = 0;
};

}
