#pragma once

#include <string>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class ReturnStatement : public Statement
{
public:
	ReturnStatement(std::string variableName) : m_variableName(variableName) {}

	std::string GetVariableName() const { return m_variableName; }

	std::string ToString() const override
	{
		return "return " + m_variableName;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_variableName;
};

}
