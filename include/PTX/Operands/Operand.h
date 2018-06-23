#pragma once

#include "PTX/Type.h"

namespace PTX {

class Operand
{
public:
	virtual std::string ToString() const = 0;
};

template<class T>
class TypedOperand : public Operand
{
	REQUIRE_BASE_TYPE(Operand, Type);
};

}
