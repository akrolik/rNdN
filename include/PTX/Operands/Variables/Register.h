#pragma once

#include "PTX/StateSpace.h"
#include "PTX/Operands/Variables/Variable.h"

namespace PTX {

template<class T>
class Variable<T, RegisterSpace> : public VariableBase<T, RegisterSpace>, public TypedOperand<T>
{
	friend class TypedVariableDeclaration<T, SpecialRegisterSpace>;
public:
	using VariableBase<T, RegisterSpace>::VariableBase;

	std::string ToString() const override
	{
		return VariableBase<T, RegisterSpace>::ToString();
	}

	json ToJSON() const override
	{
		json j = Variable<T, RegisterSpace>::ToJSON();
		j["kind"] = "PTX::Register";
		j.erase("space");
		return j;
	}
};

template<class T>
using Register = Variable<T, RegisterSpace>;

}
