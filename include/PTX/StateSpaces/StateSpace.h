#pragma once

#include "PTX/Statements/DirectiveStatement.h"
#include "PTX/Type.h"

namespace PTX {

template<class T, VectorSize V = Scalar>
class StateSpace : public DirectiveStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	virtual std::string SpaceName() = 0;
	virtual std::string Name() = 0;

	virtual std::string ToString() = 0;
};

}
