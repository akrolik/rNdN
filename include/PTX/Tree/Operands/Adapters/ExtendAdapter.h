#pragma once

#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<template<Bits> class T, Bits DB, Bits SB>
class ExtendAdapter : public Adapter<T, T, DB, SB>
{
public:
	ExtendAdapter(TypedOperand<T<SB>> *operand) : Adapter<T, T, DB, SB>(operand) {}

	json ToJSON() const override
	{
		json j = Adapter<T, T, DB, SB>::ToJSON();
		j["kind"] = "PTX::ExtendAdapter";
		return j;
	}
};

template<template<Bits> class T, Bits DB, Bits SB, class S>
class ExtendVariableAdapter : public VariableAdapter<T<DB>, T<SB>, S>
{
public:
	using VariableAdapter<T<DB>, T<SB>, S>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<T<DB>, T<SB>, S>::ToJSON();
		j["kind"] = "PTX::ExtendVariableAdapter";
		return j;
	}
};

template<template<Bits> class T, Bits DB, Bits SB>
using ExtendRegisterAdapter = ExtendVariableAdapter<T, DB, SB, RegisterSpace>;

}
