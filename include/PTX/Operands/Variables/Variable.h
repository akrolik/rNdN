#pragma once

#include "PTX/Operands/Operand.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"

namespace PTX {

template<class T, class S>
class VariableDeclaration;

template<class T, class S>
class Variable : public TypedOperand<T>
{
	friend class VariableDeclaration<T, S>;

	REQUIRE_BASE_TYPE(Variable, Type);
	REQUIRE_BASE_SPACE(Variable, StateSpace);
public:
	using VariableType = T;
	using VariableSpace = S;

	virtual std::string GetName() const
	{
		return m_name;
	}

	std::string ToString() const override
	{
		return GetName();
	}

	json ToJSON () const override
	{
		json j;
		j["kind"] = "PTX::Variable";
		j["name"] = m_name;
		j["type"] = T::Name();
		j["space"] = S::Name();
		return j;
	}

protected:
	Variable(const std::string& name) : m_name(name) {}

	std::string m_name;
};

}
