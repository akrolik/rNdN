#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Operands/Operand.h"

namespace PTX {

class InstructionStatement : public Statement
{
public:
	std::string ToString(unsigned int indentation) const override
	{
		std::string code = std::string(indentation, '\t') + OpCode();
		bool first = true;
		for (const auto& operand : Operands())
		{
			if (first)
			{
				code += " ";
				first = false;
			}
			else
			{
				code += ", ";
			}
			code += operand->ToString();
		}
		return code + ";";
	}

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

	// Visitors

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& operand : Operands())
			{
				operand->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}
};

}
