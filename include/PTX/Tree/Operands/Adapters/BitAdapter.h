#pragma once

#include "PTX/Tree/Operands/Adapters/Adapter.h"
#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"

namespace PTX {

template<template<Bits> class C, Bits B>
class BitAdapter : public Adapter<BitType, C, B>
{
public:
	BitAdapter(const TypedOperand<C<B>> *operand) : Adapter<BitType, C, B>(operand) {}
};

template<template<Bits> class C>
using Bit8Adapter = BitAdapter<C, Bits::Bits8>;
template<template<Bits> class C>
using Bit16Adapter = BitAdapter<C, Bits::Bits16>;
template<template<Bits> class C>
using Bit32Adapter = BitAdapter<C, Bits::Bits32>;
template<template<Bits> class C>
using Bit64Adapter = BitAdapter<C, Bits::Bits64>;

template<template<Bits> class C, Bits B, class S>
class BitVariableAdapter : public VariableAdapter<BitType<B>, C<B>, S>
{
public:
	using VariableAdapter<BitType<B>, C<B>, S>::VariableAdapter;

	json ToJSON() const override
	{
		json j = VariableAdapter<BitType<B>, C<B>, S>::ToJSON();
		j["kind"] = "PTX::BitVariableAdapter";
		return j;
	}
};

template<template<Bits> class C, class S>
using Bit8VariableAdapter = BitVariableAdapter<C, Bits::Bits8, S>;
template<template<Bits> class C, class S>
using Bit16VariableAdapter = BitVariableAdapter<C, Bits::Bits16, S>;
template<template<Bits> class C, class S>
using Bit32VariableAdapter = BitVariableAdapter<C, Bits::Bits32, S>;
template<template<Bits> class C, class S>
using Bit64VariableAdapter = BitVariableAdapter<C, Bits::Bits64, S>;

template<template<Bits> class C, Bits B>
using BitRegisterAdapter = BitVariableAdapter<C, B, RegisterSpace>;

template<template<Bits> class C>
using Bit8RegisterAdapter = BitRegisterAdapter<C, Bits::Bits8>;
template<template<Bits> class C>
using Bit16RegisterAdapter = BitRegisterAdapter<C, Bits::Bits16>;
template<template<Bits> class C>
using Bit32RegisterAdapter = BitRegisterAdapter<C, Bits::Bits32>;
template<template<Bits> class C>
using Bit64RegisterAdapter = BitRegisterAdapter<C, Bits::Bits64>;

}
