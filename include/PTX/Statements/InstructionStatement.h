#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Operands/Operand.h"

#include "Libraries/json.hpp"

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

	std::string Terminator() const override { return ";"; }

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::InstructionStatement";
		j["opcode"] = OpCode();
		for (const auto& operand : Operands())
		{
			j["operands"].push_back(operand->ToJSON());
		}
		return j;
	}

	virtual std::string OpCode() const = 0;
	virtual std::vector<const Operand *> Operands() const = 0;
};

}
