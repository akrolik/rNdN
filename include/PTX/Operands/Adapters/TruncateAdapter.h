#pragma once

#include "PTX/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<template<Bits> class T, Bits DB, Bits SB, class S>
class TruncateVariableAdapter : public VariableAdapter<T<DB>, T<SB>, S>
{
public:
	using VariableAdapter<T<DB>, T<SB>, S>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<T<DB>, T<SB>, S>::ToJSON();
		j["kind"] = "PTX::TruncateVariableAdapter";
		return j;
	}
};

template<template<Bits> class T, Bits DB, Bits SB>
using TruncateRegisterAdapter = TruncateVariableAdapter<T, DB, SB, RegisterSpace>;

}
