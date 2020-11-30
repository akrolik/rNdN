#pragma once

#include "PTX/Statements/Statement.h"

namespace PTX {

class DirectiveStatement : public Statement
{
public:
	// Visitors

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}
};

}
