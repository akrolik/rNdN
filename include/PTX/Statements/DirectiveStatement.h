#pragma once

#include "PTX/Statements/Statement.h"

namespace PTX {

class DirectiveStatement : public Statement
{
	std::string Terminator() const override { return ";"; }
};

}
