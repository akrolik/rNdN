#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T, class S>
class AddressableVariable : public Variable<T, S>
{
	friend class VariableDeclaration<T, S>;
public:
	REQUIRE_TYPE_PARAM(AddressableVariable,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(AddressableVariable,
		REQUIRE_BASE(S, AddressableSpace)
	);

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
