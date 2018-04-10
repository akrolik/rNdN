#pragma once

#include "PTX/Statements/DirectiveStatement.h"
#include "PTX/Type.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class StateSpace : public DirectiveStatement
{
public:
	virtual std::string SpaceName() = 0;

	virtual std::string ToString() = 0;
};

}
