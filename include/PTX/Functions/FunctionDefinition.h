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

	std::string ToString(unsigned int indentation = 0) const override
	{
		return Function::ToString(indentation) + "\n" + StatementList::ToString(0);
	}
};

}
