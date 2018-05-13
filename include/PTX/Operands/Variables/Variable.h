#pragma once

#include "PTX/Operands/Operand.h"

#include "PTX/Type.h"
#include "PTX/StateSpaces/StateSpace.h"

namespace PTX {

template<class T>
class StateSpace;

template<class T, class S>
class Variable : public Operand<T>
{
	friend S;

	static_assert(std::is_base_of<StateSpace<T>, S>::value, "S must be a PTX::StateSpace<T>");
public:
	virtual std::string GetName() const
	{
		return m_name;
	}

	virtual S *GetStateSpace() const
	{
		return m_stateSpace;
	}

	std::string ToString() const
	{
		return GetName();
	}

protected:
	Variable(std::string name, S *stateSpace) : m_name(name), m_stateSpace(stateSpace) {}

	std::string m_name;
	S *m_stateSpace = nullptr;
};


}