#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

DispatchInterface_2(DualOperand)

template<class T1, class T2, bool Assert = true>
class DualOperand : DispatchInherit(DualOperand), public Operand
{
public:
	REQUIRE_TYPE_PARAMS(DualOperand,
		REQUIRE_BASE(T1, DataType),
		REQUIRE_BASE(T2, DataType)
	)

	DualOperand(TypedOperand<T1> *operandP, TypedOperand<T2> *operandQ) : m_operandP(operandP), m_operandQ(operandQ) {}

	// Properties

	const TypedOperand<T1> *GetOperandP() const { return m_operandP; }
	TypedOperand<T1> *GetOperandP() { return m_operandP; }
	void SetOperandP(TypedOperand<T1> *operandP) { m_operandP = operandP; }

	const TypedOperand<T2> *GetOperandQ() const { return m_operandQ; }
	TypedOperand<T2> *GetOperandQ() { return m_operandQ; }
	void SetOperandQ(TypedOperand<T2> *operandQ) { m_operandQ = operandQ; }

	// Formatting

	std::string ToString() const override
	{
		return m_operandP->ToString() + "|" + m_operandQ->ToString();
	}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX:DualOperand";
		j["operand_p"] = m_operandP->ToJSON();
		j["operand_q"] = m_operandQ->ToJSON();
		return j;
	}

	void Accept(OperandVisitor& visitor) override
	{
		if (visitor.Visit(this))
		{
			m_operandP->Accept(visitor);
			m_operandQ->Accept(visitor);
		}
	}

	void Accept(ConstOperandVisitor& visitor) const override
	{
		if (visitor.Visit(this))
		{
			m_operandP->Accept(visitor);
			m_operandQ->Accept(visitor);
		}
	}

private:
	DispatchMember_Type1(T1);
	DispatchMember_Type2(T2);

	TypedOperand<T1> *m_operandP = nullptr;
	TypedOperand<T2> *m_operandQ = nullptr;
};

}
