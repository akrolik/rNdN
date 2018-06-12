#pragma once

#include "PTX/Operands/Operand.h"

#include "PTX/Resource.h"
#include "PTX/StateSpace.h"
#include "PTX/Type.h"

namespace PTX {

template<class T, class S>
class Variable : public Operand<T>, public Resource<S>
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

	std::string ToString() const
	{
		return GetName();
	}

protected:
	Variable(const std::string& name) : m_name(name) {}

	std::string m_name;
};

}
