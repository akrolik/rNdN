#pragma once

#include "PTX/Type.h"

namespace PTX {

template<class T>
class StateSpace;

template<class T, class S>
class Variable;

template<class T, class S>
class VariableSet
{
	friend S;

	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
	static_assert(std::is_base_of<StateSpace<T>, S>::value, "S must be a PTX::StateSpace<T>");
public:
	VariableSet() {}

	Variable<T, S> *GetVariable(unsigned int index)
	{
		if (index >= m_count)
		{
			std::cerr << "[Error] Variable index " << index << " out of bounds" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		return new Variable<T, S>(GetName(index), m_stateSpace);
	}

	virtual std::string GetPrefix() const
	{
		return m_name;
	}

	virtual std::string GetName(unsigned int index) const
	{
		if (index >= m_count)
		{
			std::cerr << "[Error] Variable index " << index << " out of bounds" << std::endl;
			std::exit(EXIT_FAILURE);
		}

		if (m_count > 1)
		{
			return m_name + std::to_string(index);

		}
		return m_name;
	}
	
	virtual S *GetStateSpace() const
	{
		return m_stateSpace;
	}

	std::string ToString() const
	{
		if (m_count > 1)
		{
			return m_name + "<" + std::to_string(m_count + 1) + ">";
		}
		return m_name;
	}

protected:
	VariableSet(std::string name, unsigned int count, S *stateSpace) : m_name(name), m_count(count), m_stateSpace(stateSpace) {}

	std::string m_name;
	unsigned int m_count = 0;
	S *m_stateSpace = nullptr;
};

}
