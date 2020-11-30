#pragma once

#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<template<Bits> class T, Bits B, class S>
class TypeVariableAdapter : public VariableAdapter<T<B>, BitType<B>, S>
{
public:
	using VariableAdapter<T<B>, BitType<B>, S>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<T<B>, BitType<B>, S>::ToJSON();
		j["kind"] = "PTX::TypeVariableAdapter";
		return j;
	}
};

template<template<Bits> class T, class S>
using Type8VariableAdapter = TypeVariableAdapter<T, Bits::Bits8, S>;
template<template<Bits> class T, class S>
using Type16VariableAdapter = TypeVariableAdapter<T, Bits::Bits16, S>;
template<template<Bits> class T, class S>
using Type32VariableAdapter = TypeVariableAdapter<T, Bits::Bits32, S>;
template<template<Bits> class T, class S>
using Type64VariableAdapter = TypeVariableAdapter<T, Bits::Bits64, S>;

template<template<Bits> class T, Bits B>
using TypeRegisterAdapter = TypeVariableAdapter<T, B, RegisterSpace>;

template<template<Bits> class T>
using Type8RegisterAdapter = TypeRegisterAdapter<T, Bits::Bits8>;
template<template<Bits> class T>
using Type16RegisterAdapter = TypeRegisterAdapter<T, Bits::Bits16>;
template<template<Bits> class T>
using Type32RegisterAdapter = TypeRegisterAdapter<T, Bits::Bits32>;
template<template<Bits> class T>
using Type64RegisterAdapter = TypeRegisterAdapter<T, Bits::Bits64>;

}
