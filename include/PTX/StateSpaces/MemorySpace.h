#pragma once

#include "PTX/StateSpaces/StateSpace.h"
#include "PTX/Operands/VariableSet.h"

namespace PTX {

template<class T>
class MemorySpace : public StateSpace<T>
{
public:
	MemorySpace() {}

	MemorySpace(std::string prefix, unsigned int count)
	{
		m_variables.push_back(new VariableSet<T, MemorySpace<T>>(prefix, count, this));
	}

	MemorySpace(std::string name)
	{
		m_variables.push_back(new Variable<T, MemorySpace<T>>(name, this));
	}

	MemorySpace(std::vector<std::string> names)
	{
		for (std::vector<std::string>::const_iterator it = names.begin(); it != names.end(); ++it)
		{
			m_variables.push_back(new Variable<T, MemorySpace<T>>(*it, this));
		}
	}

	virtual Variable<T, MemorySpace<T>> *GetVariable(std::string name, unsigned int index = 0) const
	{
		for (typename std::vector<VariableSet<T, MemorySpace<T>> *>::const_iterator it = m_variables.begin(); it != m_variables.end(); ++it)
		{
			if ((*it)->GetPrefix() == name)
			{
				return (*it)->GetVariable(index);
			}
		}
		std::cerr << "[Error] Variable " << name << " not found in StateSpace" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	virtual AddressSpace GetAddressSpace() const = 0;

protected:
	std::string VariableNames() const
	{
		std::ostringstream code;
		bool first = true;
		for (typename std::vector<VariableSet<T, MemorySpace<T>> *>::const_iterator it = m_variables.begin(); it != m_variables.end(); ++it)
		{
			if (!first)
			{
				code << ", ";
				first = false;
			}
			code << (*it)->ToString();
		}
		return code.str();
	}

	std::vector<VariableSet<T, MemorySpace<T>> *> m_variables;
};

}
