#pragma once

#include "PTX/Operands/Adapters/Adapter.h"
#include "PTX/Operands/Variables/Register.h"

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

template<template<Bits> class C, Bits B>
class BitRegisterAdapter : public Register<BitType<B>>
{
public:
	BitRegisterAdapter(const Register<C<B>> *variable) : Register<BitType<B>>(variable->GetName()) {}
};

template<template<Bits> class C>
using Bit8RegisterAdapter = BitRegisterAdapter<C, Bits::Bits8>;
template<template<Bits> class C>
using Bit16RegisterAdapter = BitRegisterAdapter<C, Bits::Bits16>;
template<template<Bits> class C>
using Bit32RegisterAdapter = BitRegisterAdapter<C, Bits::Bits32>;
template<template<Bits> class C>
using Bit64RegisterAdapter = BitRegisterAdapter<C, Bits::Bits64>;

}
