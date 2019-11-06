#pragma once

#include "PTX/Node.h"

#include "PTX/Statements/Statement.h"

namespace PTX {

class StatementList : public Node
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

	std::string ToString(unsigned int indentation) const override
	{
		std::string code;
		code += std::string(indentation, '\t') + "{\n";
		for (const auto& statement : m_statements)
		{
			code += statement->ToString(indentation + 1) + "\n";
		}
		return code + std::string(indentation, '\t') + "}";
	}

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
	std::vector<const Statement *> m_statements;
};

}
