#pragma once

#include "PTX/Statements/Statement.h"

#include "Libraries/json.hpp"

namespace PTX {

class StatementList
{
public:
	const std::vector<const Statement *>& GetStatements() const { return m_statements; }
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

	json ToJSON() const
	{
		json j;
		for (const auto& statement : m_statements)
		{
			j.push_back(statement->ToJSON());
		}
		return j;
	}

protected:
	std::vector<const Statement *> m_statements;
};

}
