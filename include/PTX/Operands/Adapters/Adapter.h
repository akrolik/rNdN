#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<template<Bits> class T, Bits B>
class BitAdapter;

template<template<Bits> class D, template<Bits> class S, Bits B>
class Adapter : public Operand<D<B>>
{
	friend class BitAdapter<S, B>;
public:
	std::string ToString() const { return m_operand->ToString(); }

protected:
	Adapter(const Operand<S<B>> *operand) : m_operand(operand) {}

	const Operand<S<B>> *m_operand = nullptr;
};

}
