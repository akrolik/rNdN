#pragma once

#include "PTX/StateSpace.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class MemorySpace : public StateSpace<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	virtual std::string SpaceName() = 0;
	virtual std::string Name() = 0;

	virtual std::string ToString() = 0;
};

}
