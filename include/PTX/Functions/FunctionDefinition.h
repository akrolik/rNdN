#pragma once

#include "PTX/Functions/FunctionDeclaration.h"

#include "PTX/Statements/StatementList.h"
#include "PTX/Statements/Statement.h"

namespace PTX {

template<class R>
class FunctionDefinition : public FunctionDeclaration<R>, public StatementList
{
public:
	json ToJSON() const override
	{
		json j = FunctionDeclaration<R>::ToJSON();
		j["statements"] = StatementList::ToJSON();
		return j;
	}

private:
	std::string Body() const override
	{
		return StatementList::ToString(0);
	}
};

}
