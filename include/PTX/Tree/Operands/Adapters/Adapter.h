#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

template<template<Bits> class D, template<Bits> class S, Bits BD, Bits BS = BD>
class Adapter : public TypedOperand<D<BD>>
{
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

	// Visitors

	void Accept(OperandVisitor& visitor) override { m_operand->Accept(visitor); }
	void Accept(ConstOperandVisitor& visitor) const override { m_operand->Accept(visitor); }

protected:
	Adapter(TypedOperand<S<BS>> *operand) : m_operand(operand) {}

	TypedOperand<S<BS>> *m_operand = nullptr;
};

}
