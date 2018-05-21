#pragma once

#include "PTX/Type.h"

namespace PTX {

template<class T>
class Operand
{
	REQUIRE_TYPE(Operand, Type);
public:
	virtual std::string ToString() const = 0;
};

}
