#pragma once

#include "PTX/Statements/Statement.h"

namespace PTX {

class BlankStatement : public Statement
{
public:
	BlankStatement() {}

	std::string ToString(unsigned int indentation = 0) const override
	{
		return "";
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::BlankStatement";
		return j;
	}
};

}
