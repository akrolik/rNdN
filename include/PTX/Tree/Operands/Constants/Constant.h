#pragma once

#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

DispatchInterface(Constant)

template<class T, bool Assert = true>
class Constant : DispatchInherit(Constant), public TypedOperand<T, Assert>
{
public:
	REQUIRE_TYPE_PARAM(Constant,
		REQUIRE_BASE(T, ScalarType)
	);

	Constant(const std::string& name) : m_name(name) {}

	// Properties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Formatting

	std::string ToString() const override
	{
		return m_name;
	}

	json ToJSON() const override
	{
		json j;
		j["type"] = T::Name();
		j["constant"] = m_name;
		return j;
	}

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	std::string m_name;
};

}
