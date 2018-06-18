#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

class BlockStatement : public Statement
{
public:
	void AddStatement(const Statement *statement)
	{
		m_statements.push_back(statement);
	}

	std::string ToString() const override
	{
		std::ostringstream code;
		code << "{" << std::endl;
		for (const auto& statement : m_statements)
		{
			code << "\t\t" << statement->ToString() << statement->Terminator() << std::endl;
		}
		code << "\t}";
		return code.str();
	}

	std::string Terminator() const override{ return ""; }
private:
	std::vector<const Statement *> m_statements;
};

}
