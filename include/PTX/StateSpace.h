#pragma once

#include "PTX/DirectiveStatement.h"

namespace PTX {

template<typename T>
class StateSpace : public DirectiveStatement
{
public:
	virtual std::string SpaceName() = 0;
	virtual std::string GetName() = 0;
};

}
