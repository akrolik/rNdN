#pragma once

#include "PTX/Operands/Operand.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"

namespace PTX {

template<class T, class S>
class TypedVariableDeclaration;

template<class T, class S>
class VariableBase : public TypedOperand<T>
{
	friend class TypedVariableDeclaration<T, S>;

	REQUIRE_TYPE_PARAM(Variable,
		REQUIRE_BASE(T, Type)
	);
	REQUIRE_SPACE_PARAM(Variable,
		REQUIRE_BASE(S, StateSpace)
	);	
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
	VariableBase(const std::string& name) : m_name(name) {}

	std::string m_name;
};

template<class T, class S, typename Enabled = void>
class Variable : public VariableBase<T, S>
{
public:
	using VariableBase<T, S>::VariableBase;
};

}
