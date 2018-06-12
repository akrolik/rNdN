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

	const std::vector<Statement *>& GetStatements() const { return m_statements; }
	void AddStatement(Statement *statement)
	{
		m_statements.push_back(statement);
	}

	std::string ToString() const
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
	std::vector<Statement *> m_statements;
};

}
