#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T, class S>
class AddressableVariable : public Variable<T, S>
{
	friend class VariableDeclaration<T, S>;

	REQUIRE_BASE_TYPE(AddressableVariable, DataType);
	REQUIRE_BASE_SPACE(AddressableVariable, AddressableSpace);
public:
	using Variable<T, S>::Variable;

	json ToJSON() const override
	{
		json j = Variable<T, S>::ToJSON();
		j["kind"] = "PTX::AddressableVariable";
		return j;
	}
};

template<class T>
using ParameterVariable = AddressableVariable<T, ParameterSpace>;

}
