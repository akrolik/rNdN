#pragma once

#include "PTX/Operands/Operand.h"

#include "PTX/Type.h"
#include "PTX/StateSpace.h"

namespace PTX {

template<class T, class S>
class Variable : public Operand<T>
{
	friend class VariableDeclaration<T, S>;
	//TODO: remove this friend class
	friend class VariableDeclaration<T, SpecialRegisterSpace>;

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
	Variable(std::string name) : m_name(name) {}

	std::string m_name;
};

}
