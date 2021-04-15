#pragma once

#include "SASS/Tree/Variable.h"

namespace SASS {

class GlobalVariable : public Variable
{
public:
	using Variable::Variable;

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	
private:
	std::string SpaceName() const override { return "global"; }
};

}
