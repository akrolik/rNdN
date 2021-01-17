#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

DispatchInterface(InvertedOperand)

template<class T, bool Assert = true>
class InvertedOperand : DispatchInherit(InvertedOperand), public TypedOperand<T, Assert>
{
public:
	REQUIRE_TYPE_PARAM(InvertedOperand,
		REQUIRE_EXACT(T, PredicateType)
	);

	InvertedOperand(TypedOperand<T> *operand) : m_operand(operand) {}

	// Properties

	const TypedOperand<T> *GetOperand() const { return m_operand; }
	TypedOperand<T> *GetOperand() { return m_operand; }
	void SetOperand(TypedOperand<T> *operand) { m_operand = operand; }

	// Formatting

	std::string ToString() const override
	{
		return "!" + m_operand->ToString();
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::InvertedOperand";
		j["operand"] = m_operand->ToJSON();
		return j;
	}

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	TypedOperand<T> *m_operand = nullptr;
};

}
