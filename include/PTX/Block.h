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

	std::string ToString() const
	{
		std::ostringstream code;
		for (std::vector<Statement *>::const_iterator it = m_statements.cbegin(); it != m_statements.cend(); it++)
		{
			Statement *stmt = *it;
			code << "\t" << stmt->ToString() << stmt->Terminator() << std::endl;
		}
		return code.str();
	}

private:
	std::vector<Statement *> m_statements;
};

}
