#pragma once

#include "SASS/Tree/Variable.h"

namespace SASS {

class SharedVariable : public Variable
{
public:
	using Variable::Variable;

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
};

}
