#pragma once

#include "PTX/Type.h"

namespace PTX {

template<Type T, VectorSize = Scalar>
class Operand
{
public:
	virtual std::string ToString() = 0;
};

}
