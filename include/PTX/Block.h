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
		for (auto it = m_statements.cbegin(); it != m_statements.cend(); it++)
		{
			code << "\t" << (*it)->ToString() << (*it)->Terminator() << std::endl;
		}
		return code.str();
	}

private:
	std::vector<Statement *> m_statements;
};

}
