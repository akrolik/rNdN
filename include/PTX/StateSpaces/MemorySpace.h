#pragma once

#include "PTX/StateSpace.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class MemorySpace : public StateSpace<T, V>
{
public:
	virtual std::string SpaceName() = 0;
	virtual std::string Name() = 0;

	virtual std::string ToString() = 0;
};

}
