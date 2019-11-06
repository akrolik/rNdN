#pragma once

#include "PTX/Type.h"
#include "PTX/Node.h"

namespace PTX {

class Operand : public Node
{
public:
	std::string ToString(unsigned int indentation) const override
	{
		return ToString();
	}

	virtual std::string ToString() const = 0;
};

template<class T>
class TypedOperand : public virtual Operand
{
	REQUIRE_TYPE_PARAM(Operand,
		REQUIRE_BASE(T, Type)
	);
};

}
