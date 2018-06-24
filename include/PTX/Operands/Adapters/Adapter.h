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

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::Adapter";
		j["destination"] = D<BD>::Name();
		j["source"] = S<BS>::Name();
		j["operand"] = m_operand->ToJSON();
		return j;
	}

protected:
	Adapter(const TypedOperand<S<BS>> *operand) : m_operand(operand) {}

	const TypedOperand<S<BS>> *m_operand = nullptr;
};

}
