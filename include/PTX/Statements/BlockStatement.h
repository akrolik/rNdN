#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Statements/StatementList.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

class BlockStatement : public Statement, public StatementList
{
public:
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

	std::string Terminator() const override { return ""; }

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::BlockStatement";
		j["statements"] = StatementList::ToJSON();
		return j;
	}
};

}
