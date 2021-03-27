#pragma once

#include "SASS/Variable.h"

namespace SASS {

class SharedVariable : public Variable
{
public:
	using Variable::Variable;

private:
	std::string SpaceName() const override { return "shared"; }
};

}
