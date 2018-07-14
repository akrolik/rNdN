#pragma once

#include "PTX/Functions/FunctionDefinition.h"

#include "PTX/Statements/StatementList.h"
#include "PTX/Statements/Statement.h"

namespace PTX {

template<class R>
class FunctionDeclaration : public FunctionDefinition<R>, public StatementList
{
public:
	json ToJSON() const override
	{
		json j = FunctionDefinition<R>::ToJSON();
		j["statements"] = StatementList::ToJSON();
		return j;
	}

private:
	std::string Body() const override
	{
		std::ostringstream code;
		code << std::endl << "{" << std::endl;
		for (const auto& statement : m_statements)
		{
			code << "\t" << statement->ToString() << statement->Terminator() << std::endl;
		}
		code << "}";
		return code.str();
	}
};

}
