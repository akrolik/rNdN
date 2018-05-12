#pragma once

#include <sstream>
#include <vector>

#include "PTX/Statements/Statement.h"

namespace PTX {

class Block
{
public:
	void AddStatement(Statement *statement)
	{
		m_statements.push_back(statement);
	}

	std::string ToString()
	{
		std::ostringstream code;
		for (std::vector<Statement *>::iterator it = m_statements.begin(); it != m_statements.end(); it++)
		{
			Statement *stmt = *it;
			code << stmt->ToString() << ";" << std::endl;
		}
		return code.str();
	}

private:
	std::vector<Statement *> m_statements;
};

}
