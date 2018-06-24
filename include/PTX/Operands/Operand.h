#pragma once

#include "PTX/Type.h"

#include "Libraries/json.hpp"

namespace PTX {

class Operand
{
public:
	virtual std::string ToString() const = 0;
	virtual json ToJSON() const = 0;
};

template<class T>
class TypedOperand : public Operand
{
	REQUIRE_BASE_TYPE(Operand, Type);
};

}
