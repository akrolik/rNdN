#pragma once

#include <string>

#include "HorseIR/Tree/Statements/Statement.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class ReturnStatement : public Statement
{
public:
	ReturnStatement(std::string identifier) : m_identifier(identifier) {}

	std::string GetIdentifier() const { return m_identifier; }

	std::string ToString() const override
	{
		return "return " + m_identifier;
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	std::string m_identifier;
};

}
