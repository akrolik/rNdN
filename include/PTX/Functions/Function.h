#pragma once

#include <string>

#include "PTX/Declarations/Declaration.h"

#include "PTX/Statements/Statement.h"

namespace PTX {

class Function : public Declaration
{
public:
	void SetName(const std::string& name) { m_name = name; }
	std::string GetName() const { return m_name; }

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

	std::string ToString() const override
	{
		std::ostringstream code;

		if (m_linkDirective != LinkDirective::None)
		{
			code << LinkDirectiveString(m_linkDirective) << " ";
		}
		code << GetDirectives() << " ";

		std::string ret = GetReturnString();
		if (ret.length() > 0)
		{
			code << "(" << ret << ") ";
		}
		code << m_name << "(" << GetParametersString() << ")" << std::endl << "{" << std::endl;

		for (const auto& statement : m_statements)
		{
			code << "\t" << statement->ToString() << statement->Terminator() << std::endl;
		}
		code << "}" << std::endl;

		return code.str();
	}

private:
	virtual std::string GetDirectives() const = 0;
	virtual std::string GetReturnString() const = 0;
	virtual std::string GetParametersString() const = 0;

	std::string m_name;
	std::vector<const Statement *> m_statements;
};

}
