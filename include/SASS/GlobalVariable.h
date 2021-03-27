#pragma once

#include "SASS/Variable.h"

namespace SASS {

class GlobalVariable : public Variable
{
public:
	using Variable::Variable;

private:
	std::string SpaceName() const override { return "global"; }
};

}
