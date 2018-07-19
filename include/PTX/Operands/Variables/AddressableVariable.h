#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T, class S>
class Variable<T, S, std::enable_if_t<std::is_base_of<AddressableSpace, S>::value>> : public VariableBase<T, S>
{
	friend class TypedVariableDeclaration<T, S>;
public:
	REQUIRE_TYPE_PARAM(AddressableVariable,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(AddressableVariable,
		REQUIRE_BASE(S, AddressableSpace)
	);

	using VariableBase<T, S>::VariableBase;

	json ToJSON() const override
	{
		json j = Variable<T, S>::ToJSON();
		j["kind"] = "PTX::AddressableVariable";
		return j;
	}
};

template<class T>
using ParameterVariable = Variable<T, ParameterSpace>;

}
