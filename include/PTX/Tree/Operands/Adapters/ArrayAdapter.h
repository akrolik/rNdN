#pragma once

#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<class T, unsigned int N, class S>
class ArrayVariableAdapter : public VariableAdapter<T, ArrayType<T, N>, S>
{
public:
	using VariableAdapter<T, ArrayType<T, N>, S>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<T, ArrayType<T, N>, S>::ToJSON();
		j["kind"] = "PTX::ArrayVariableAdapter";
		return j;
	}
};

}
