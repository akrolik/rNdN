#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T>
class Register : public Variable<T, RegisterSpace>
{
	friend class VariableDeclaration<T, SpecialRegisterSpace>;
public:
	using Variable<T, RegisterSpace>::Variable;

	json ToJSON() const override
	{
		json j = Variable<T, RegisterSpace>::ToJSON();
		j["kind"] = "PTX::Register";
		j.erase("space");
		return j;
	}
};

}
