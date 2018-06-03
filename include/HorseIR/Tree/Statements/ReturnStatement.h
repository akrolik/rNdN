#pragma once

#include "HorseIR/Tree/Statements/Statement.h"

namespace HorseIR {

class ReturnStatement : public Statement
{
public:
	ReturnStatement(std::string identifier) : m_identifier(identifier) {}

	std::string ToString() const
	{
		return "return " + m_identifier;
	}

private:
	std::string m_identifier;
};

}
