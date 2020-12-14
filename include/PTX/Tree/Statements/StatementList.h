#pragma once

#include "PTX/Tree/Node.h"

#include "PTX/Tree/Statements/Statement.h"

namespace PTX {

class StatementList : public Node
{
public:
	// Properties

	std::vector<const Statement *> GetStatements() const
	{
		return { std::begin(m_statements), std::end(m_statements) };
	}
	std::vector<Statement *>& GetStatements() { return m_statements; }

	void AddStatement(Statement *statement)
	{
		m_statements.push_back(statement);
	}
	template<class T>
	void AddStatements(const std::vector<T>& statements)
	{
		m_statements.insert(std::end(m_statements), std::begin(statements), std::end(statements));
	}

	void InsertStatement(Statement *statement, unsigned int index)
	{
		m_statements.insert(std::begin(m_statements) + index, statement);
	}
	template<class T>
	void InsertStatements(const std::vector<T>& statements, unsigned int index)
	{
		m_statements.insert(std::begin(m_statements) + index, std::begin(statements), std::end(statements));
	}

	// Formatting

	json ToJSON() const override
	{
		json j;
		for (const auto& statement : m_statements)
		{
			j.push_back(statement->ToJSON());
		}
		return j;
	}

protected:
	std::vector<Statement *> m_statements;
};

}
