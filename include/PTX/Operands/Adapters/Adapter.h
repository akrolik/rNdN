#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<template<Bits> class T, Bits B>
class BitAdapter;

template<template<Bits> class D, template<Bits> class S, Bits BD, Bits BS = BD>
class Adapter : public TypedOperand<D<BD>>
{
	friend class BitAdapter<S, BD>;
public:
	std::string ToString() const { return m_operand->ToString(); }

protected:
	Adapter(const TypedOperand<S<BS>> *operand) : m_operand(operand) {}

	const TypedOperand<S<BS>> *m_operand = nullptr;
};

}
