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
	template<class T>
	void AddStatements(const std::vector<T>& statements)
	{
		m_statements.insert(std::end(m_statements), std::begin(statements), std::end(statements));
	}

	void InsertStatement(const Statement *statement, unsigned int index)
	{
		m_statements.insert(std::begin(m_statements) + index, statement);
	}
	template<class T>
	void InsertStatements(const std::vector<T>& statements, unsigned int index)
	{
		m_statements.insert(std::begin(m_statements) + index, std::begin(statements), std::end(statements));
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
